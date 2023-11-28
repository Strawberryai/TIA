# Configuraciones de hiperparámetros y arquitecturas

Una capa oculta con 5 neuronas:

self.batch_size = 20
self.lr = -0.01
self.w0 = nn.Parameter(1, 5)
self.b0 = nn.Parameter(1, 5)
self.w1 = nn.Parameter(5, 1)
self.b1 = nn.Parameter(1, 1)