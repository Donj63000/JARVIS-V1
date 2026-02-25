use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

const BM25_K1: f32 = 1.5;
const BM25_B: f32 = 0.75;

#[derive(Clone, Debug)]
pub struct GuideChunk {
    pub id: usize,
    pub section: String,
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
    token_counts: HashMap<String, usize>,
    token_len: usize,
}

#[derive(Clone, Debug)]
pub struct RetrievalHit {
    pub chunk_id: usize,
    pub section: String,
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
    pub score: f32,
}

#[derive(Clone, Debug)]
pub struct GuideKnowledgeBase {
    source_path: PathBuf,
    chunks: Vec<GuideChunk>,
    doc_freqs: HashMap<String, usize>,
    avg_doc_len: f32,
}

#[derive(Clone, Debug)]
struct Paragraph {
    section: String,
    line_start: usize,
    line_end: usize,
    text: String,
}

impl GuideKnowledgeBase {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).with_context(|| {
            format!(
                "Impossible de lire le guide de production: {}",
                path.display()
            )
        })?;

        let paragraphs = parse_paragraphs(&content);
        let chunks = build_chunks(&paragraphs, 1300);
        let (doc_freqs, avg_doc_len) = compute_corpus_stats(&chunks);

        Ok(Self {
            source_path: path.to_path_buf(),
            chunks,
            doc_freqs,
            avg_doc_len,
        })
    }

    pub fn try_load_default() -> Option<Self> {
        let candidate = Path::new("guide-production-rochias.txt");
        if !candidate.exists() {
            return None;
        }
        Self::load(candidate).ok()
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<RetrievalHit> {
        let query_tokens = tokenize_for_search(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| (index, bm25_score(self, chunk, &query_tokens)))
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        scored
            .into_iter()
            .take(top_k)
            .map(|(index, score)| {
                let chunk = &self.chunks[index];
                RetrievalHit {
                    chunk_id: chunk.id,
                    section: chunk.section.clone(),
                    line_start: chunk.line_start,
                    line_end: chunk.line_end,
                    text: chunk.text.clone(),
                    score,
                }
            })
            .collect()
    }
}

fn bm25_score(index: &GuideKnowledgeBase, chunk: &GuideChunk, query_tokens: &[String]) -> f32 {
    let mut score = 0.0;
    let doc_len = chunk.token_len as f32;
    let avg_len = index.avg_doc_len.max(1.0);

    for token in query_tokens {
        let freq = *chunk.token_counts.get(token).unwrap_or(&0) as f32;
        if freq == 0.0 {
            continue;
        }

        let df = *index.doc_freqs.get(token).unwrap_or(&0) as f32;
        let n_docs = index.chunks.len() as f32;
        let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

        let numerator = freq * (BM25_K1 + 1.0);
        let denominator = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avg_len));
        score += idf * (numerator / denominator);
    }

    score
}

fn compute_corpus_stats(chunks: &[GuideChunk]) -> (HashMap<String, usize>, f32) {
    let mut doc_freqs: HashMap<String, usize> = HashMap::new();
    let mut total_len = 0usize;

    for chunk in chunks {
        total_len += chunk.token_len;
        for token in chunk.token_counts.keys() {
            *doc_freqs.entry(token.clone()).or_insert(0) += 1;
        }
    }

    let avg_doc_len = if chunks.is_empty() {
        0.0
    } else {
        total_len as f32 / chunks.len() as f32
    };

    (doc_freqs, avg_doc_len)
}

fn parse_paragraphs(content: &str) -> Vec<Paragraph> {
    let mut paragraphs = Vec::new();
    let mut current_section = "Introduction".to_string();
    let mut current_lines: Vec<(usize, String)> = Vec::new();

    for (idx, raw_line) in content.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();

        if line.is_empty() {
            flush_paragraph(&mut paragraphs, &mut current_lines, &current_section);
            continue;
        }

        if is_section_heading(line) {
            flush_paragraph(&mut paragraphs, &mut current_lines, &current_section);
            current_section = line.to_string();
            continue;
        }

        current_lines.push((line_no, line.to_string()));
    }

    flush_paragraph(&mut paragraphs, &mut current_lines, &current_section);
    paragraphs
}

fn flush_paragraph(out: &mut Vec<Paragraph>, lines: &mut Vec<(usize, String)>, section: &str) {
    if lines.is_empty() {
        return;
    }

    let line_start = lines.first().map(|(line_no, _)| *line_no).unwrap_or(1);
    let line_end = lines
        .last()
        .map(|(line_no, _)| *line_no)
        .unwrap_or(line_start);
    let text = lines
        .iter()
        .map(|(_, line)| line.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    out.push(Paragraph {
        section: section.to_string(),
        line_start,
        line_end,
        text,
    });
    lines.clear();
}

fn build_chunks(paragraphs: &[Paragraph], max_chars: usize) -> Vec<GuideChunk> {
    let mut chunks = Vec::new();
    let mut current_text = String::new();
    let mut current_section = String::new();
    let mut current_start = 1usize;
    let mut current_end = 1usize;
    let mut next_chunk_id = 1usize;

    for paragraph in paragraphs {
        if current_text.is_empty() {
            current_section = paragraph.section.clone();
            current_start = paragraph.line_start;
            current_end = paragraph.line_end;
            current_text = paragraph.text.clone();
            continue;
        }

        let candidate_len = current_text.len() + 1 + paragraph.text.len();
        if candidate_len <= max_chars {
            current_text.push(' ');
            current_text.push_str(&paragraph.text);
            current_end = paragraph.line_end;
        } else {
            chunks.push(new_chunk(
                next_chunk_id,
                &current_section,
                current_start,
                current_end,
                &current_text,
            ));
            next_chunk_id += 1;

            current_section = paragraph.section.clone();
            current_start = paragraph.line_start;
            current_end = paragraph.line_end;
            current_text = paragraph.text.clone();
        }
    }

    if !current_text.is_empty() {
        chunks.push(new_chunk(
            next_chunk_id,
            &current_section,
            current_start,
            current_end,
            &current_text,
        ));
    }

    chunks
}

fn new_chunk(
    id: usize,
    section: &str,
    line_start: usize,
    line_end: usize,
    text: &str,
) -> GuideChunk {
    let tokens = tokenize_for_search(text);
    let mut token_counts: HashMap<String, usize> = HashMap::new();
    for token in &tokens {
        *token_counts.entry(token.clone()).or_insert(0) += 1;
    }

    GuideChunk {
        id,
        section: section.to_string(),
        line_start,
        line_end,
        text: text.to_string(),
        token_counts,
        token_len: tokens.len(),
    }
}

fn is_section_heading(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.len() < 3 {
        return false;
    }

    let letter_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    if letter_count < 3 {
        return false;
    }

    let uppercase_count = trimmed
        .chars()
        .filter(|c| c.is_alphabetic() && c.is_uppercase())
        .count();

    let uppercase_ratio = uppercase_count as f32 / letter_count as f32;
    uppercase_ratio > 0.75
}

pub fn normalize_for_search(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.nfkd() {
        if is_combining_mark(c) {
            continue;
        }

        let mapped = if c.is_ascii_alphanumeric() || c.is_alphabetic() || c.is_numeric() {
            c.to_ascii_lowercase()
        } else {
            ' '
        };

        if mapped.is_alphabetic() || mapped.is_numeric() {
            out.push(mapped);
        } else {
            out.push(' ');
        }
    }
    out
}

pub fn tokenize_for_search(text: &str) -> Vec<String> {
    normalize_for_search(text)
        .split_whitespace()
        .map(|token| token.to_string())
        .collect()
}

pub fn build_grounded_system_prompt(base_system: &str) -> String {
    format!(
        "{base_system}\n\nRegles de fiabilite obligatoires:\n- Reponds uniquement a partir des extraits du guide fournis.\n- Si l'information n'est pas dans les extraits, reponds exactement: \"Information non trouvee dans le guide fourni.\"\n- N'invente jamais de procedure, valeur, ou consigne.\n- Chaque affirmation factuelle doit citer au moins une reference au format [REF n] (exemple: [REF 2])."
    )
}

pub fn build_grounded_user_prompt(question: &str, hits: &[RetrievalHit]) -> String {
    let mut context = String::new();
    for hit in hits {
        context.push_str(&format!(
            "[REF {}] section=\"{}\" lignes=\"L{}-L{}\" score=\"{:.3}\"\n{}\n\n",
            hit.chunk_id, hit.section, hit.line_start, hit.line_end, hit.score, hit.text
        ));
    }

    if context.trim().is_empty() {
        context.push_str("Aucun extrait pertinent n'a ete retrouve.\n");
    }

    format!(
        "Question utilisateur:\n{question}\n\nExtraits du guide:\n{context}\nContraintes:\n- Reponds uniquement a partir des extraits.\n- Cite les sources uniquement avec [REF n] (n est le numero de reference ci-dessus).\n- Si les extraits ne suffisent pas: \"Information non trouvee dans le guide fourni.\""
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn normalize_for_search_removes_accents() {
        let s = normalize_for_search("Débit séchoir Équipe");
        assert_eq!(
            s.split_whitespace().collect::<Vec<_>>(),
            vec!["debit", "sechoir", "equipe"]
        );
    }

    #[test]
    fn search_finds_expected_chunk() {
        let guide = "\
TITRE\n\
\n\
LA SALLE DE CONTROLE\n\
\n\
Le gyrophare rouge emet un son continu en cas de defaut.\n\
\n\
AUTRE\n\
\n\
Le fournisseur est renseigne dans la saisie de production.\n";

        let mut tmp = std::env::temp_dir();
        tmp.push("guide-test-rochias.txt");
        fs::write(&tmp, guide).unwrap();

        let kb = GuideKnowledgeBase::load(&tmp).unwrap();
        let hits = kb.search("Que signifie le gyrophare rouge ?", 3);

        assert!(!hits.is_empty());
        assert!(hits[0].text.to_lowercase().contains("gyrophare"));
    }

    #[test]
    fn grounded_prompt_contains_citations() {
        let hits = vec![RetrievalHit {
            chunk_id: 1,
            section: "SALLE".to_string(),
            line_start: 10,
            line_end: 12,
            text: "Texte test".to_string(),
            score: 1.0,
        }];

        let user_prompt = build_grounded_user_prompt("Question ?", &hits);
        assert!(user_prompt.contains("[REF 1]"));
        assert!(user_prompt.contains("section=\"SALLE\""));
        assert!(user_prompt.contains("lignes=\"L10-L12\""));
        assert!(user_prompt.contains("Information non trouvee"));
    }
}
