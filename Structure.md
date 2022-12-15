- Cover
- Roadmap: (General Architecture, Comparison of Methods, Future Considerarions)
- General Architecture: Preprocessing -> Feature extraction (Train & Test) -> training a Multi-layer perpcepton -> doing inference over test set -> computing pearson correlation
- Preprocessing: 
          We coded the test_processing class
          Methods included:
                - clean_data(lowercase, stopwords, shortwords, signs) (using in the final system)
                - tokenize_data()
                - frequency ()
                - lemmatize_data(use_pos_tag=True) (using in the final system)
                - most_common_lemma_data()
                - apply_lesk()
                - get_named_entities
- Feature Extraction:
          Same for train and test data
          We coded the compute_metrics class
          Methods included:
                - jaccard_similarity() (using)     -> Lemmas, semi_clean_data, clean_data
                - normalized_length_difference()
                - synsets_similarities(Lin)(using) -> Lemmas
                - synsets_similarities(Lch)(using) -> Lemmas
                - synsets_similarities(Path)(using)-> Lemmas
                - synsets_similarities(Wup)
                - cosine_similarity()(using) 
                - unigrams_similarity()(using)
                - bigrams_similarity()(using)
                - trigrams_similarity()(using)
- Traning of MLP
            - Precomputed feature extraction of train data and stored it in npy files for efficency
            - 4 Hidden layers
            - (100,100,100,100) neurons in each layer
            - Max. iters = 200 
            - Pass the extracted features of train data and the gold-standandard scores as labels
- Inference (Test_data)
      - Precomputed feature extraction of test data and stored it in npy files for efficency
      - Predict with the MLP
- Results -> comparison of methods
