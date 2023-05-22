from sentence_transformers import SentenceTransformer

# load model
transformer_model = SentenceTransformer('all-mpnet-base-v2')

# save model
transformer_model.save('all-mpnet-base-v2')