encoder_config = {
    'input_dim': 2,
    'hidden_dim': 128,
    'num_gat_layers': 3,
    'num_heads': 1,
    'gat_p_dropout': 0.5,
    'loc_embed_intermediate_dim': 64,
    'loc_embed_p_dropout': 0.1, 
}

training_config = {
    'learning_rate': 1e-4,
    'num_epochs': 1000, 
    'batch_size': 512,
    'train_data_size': 100_000,
}

val_config = {
    'num_cities': 20,
    'num_instances': 20, 
}
