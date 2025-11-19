def test_import():
    from src.aetherquery.embeddings import create_embeddings
    from src.aetherquery.llm import create_flan_t5_pipeline

    emb = create_embeddings()
    pipe = create_flan_t5_pipeline(load_in_8bit=False)

    assert emb is not None
    assert pipe is not None