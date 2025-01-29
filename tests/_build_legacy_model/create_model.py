import numpy as np

import cebra

neural_data = np.random.normal(0, 1, (1000, 30))  # 1000 samples, 30 features
cebra_model = cebra.CEBRA(model_architecture="offset10-model",
                          batch_size=512,
                          learning_rate=1e-4,
                          max_iterations=10,
                          time_offsets=10,
                          num_hidden_units=16,
                          output_dimension=8,
                          verbose=True)
cebra_model.fit(neural_data)
cebra_model.save("cebra_model.pt")
