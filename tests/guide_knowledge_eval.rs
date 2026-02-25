use localai::guide_knowledge::{
    build_grounded_system_prompt, build_grounded_user_prompt, normalize_for_search,
    GuideKnowledgeBase,
};
use std::fs;
use std::path::Path;

#[test]
fn retrieval_hits_expected_facts_in_real_guide() {
    let guide_path = Path::new("guide-production-rochias.txt");
    if !guide_path.exists() {
        // Local-only dataset: skip when absent.
        return;
    }

    let kb = GuideKnowledgeBase::load(guide_path).unwrap();

    let cases = vec![
        (
            "Quel signal apparait en cas de defaut dans la salle de controle ?",
            "gyrophare rouge",
        ),
        (
            "Combien de cellules sont mentionnees pour le sechoir ?",
            "huit cellules",
        ),
        (
            "Peut-on demarrer le tapis 2 si le tapis 3 n'est pas demarre ?",
            "ne pouvez pas demarrer le tapis 2",
        ),
    ];

    for (query, expected) in cases {
        let hits = kb.search(query, 5);
        assert!(!hits.is_empty(), "Aucun hit pour la question: {query}");

        let expected_norm = normalize_for_search(expected);
        let found = hits.iter().any(|hit| {
            let hit_norm = normalize_for_search(&hit.text);
            hit_norm.contains(expected_norm.trim())
        });

        assert!(
            found,
            "La reponse attendue n'est pas dans les hits pour la question: {query}"
        );
    }
}

#[test]
fn grounded_prompt_has_guardrails_and_citations() {
    let guide = "\
SECTION SECURITE\n\
\n\
Le gyrophare rouge emet un son continu en cas de defaut.\n";

    let mut tmp = std::env::temp_dir();
    tmp.push("guide-prompt-guardrails.txt");
    fs::write(&tmp, guide).unwrap();

    let kb = GuideKnowledgeBase::load(&tmp).unwrap();
    let hits = kb.search("Que signifie le gyrophare rouge ?", 3);
    let user_prompt = build_grounded_user_prompt("Que signifie le gyrophare rouge ?", &hits);
    let system_prompt = build_grounded_system_prompt("Tu es assistant de production.");

    assert!(user_prompt.contains("Extraits du guide"));
    assert!(user_prompt.contains("[REF"));
    assert!(user_prompt.contains("lignes=\"L"));
    assert!(system_prompt.contains("Regles de fiabilite obligatoires"));
    assert!(system_prompt.contains("Information non trouvee dans le guide fourni."));
}
