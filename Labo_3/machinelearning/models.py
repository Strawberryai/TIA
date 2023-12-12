import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.
        
        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        Deberiais obtener el producto escalar (o producto punto) que es "equivalente" a la distancia del coseno
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)



    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Dependiendo del valor del coseno devolvera 1 o -1
        
        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if ( nn.as_scalar(self.run(x)) >= 0 ) else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Hasta que TODOS los ejemplos del train esten bien clasificados. Es decir, hasta que la clase predicha en se corresponda con la real en TODOS los ejemplos del train
        """
        "*** YOUR CODE HERE ***"
        
        converge = False
        while not converge:
            # Inicio de epoch
            cambio = False
            
            for x, y in dataset.iterate_once(1):
                # x es el vector de entrada e y es el gold
                pred = self.get_prediction(x)
                
                y = nn.as_scalar(y)
                if pred != y:
                    cambio = True
                    self.w.update( x, y )
            
            # Fin de epoch
            if not cambio:
                converge = True
        

        # Fin de entrenamiento
        return



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    NO ES CLASIFICACION, ES REGRESION. ES DECIR; APRENDER UNA FUNCION.
    SI ME DAN X TENGO QUE APRENDER A OBTENER LA MISMA Y QUE EN LA FUNCION ORIGINAL DE LA QUE QUIERO APRENDER
    """
    def __init__(self):
        # Initialize your model parameters here
        # For example:
        self.batch_size = 20
        self.lr = -0.01
        self.w0 = nn.Parameter(1, 12)
        self.b0 = nn.Parameter(1, 12)
        
        self.w1 = nn.Parameter(12, 50)
        self.b1 = nn.Parameter(1, 50)
        
        self.w2 = nn.Parameter(50, 1)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1). En este caso cada ejemplo solo esta compuesto por un rasgo
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values.
            Como es un modelo de regresion, cada valor y tambien tendra un unico valor
        """
        "*** YOUR CODE HERE ***"
        #return nn.AddBias(nn.ReLU(nn.Linear(nn.AddBias(nn.ReLU(nn.Linear(x, self.w0)), self.b0), self.w1)), self.b1)
    
        o1 = nn.Linear(x, self.w0)
        o1 = nn.AddBias(o1, self.b0)
        o1 = nn.ReLU( o1 )
        
        
        o2 = nn.Linear(o1, self.w1)
        o2 = nn.AddBias(o2, self.b1)
        o2 = nn.ReLU( o2 )
        
        o3 = nn.Linear(o2, self.w2)
        o3 = nn.AddBias(o3, self.b2)
        #o3 = nn.ReLU( o3 )
        
        return o3
        


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
                ----> ES FACIL COPIA Y PEGA ESTO Y ANNADE LA VARIABLE QUE HACE FALTA PARA CALCULAR EL ERROR 
                return nn.SquareLoss(self.run(x),ANNADE LA VARIABLE QUE ES NECESARIA AQUI), para medir el error, necesitas comparar el resultado de tu prediccion con .... que?
        """
        "*** YOUR CODE HERE ***"       
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        
        """
        
        total_loss = 100000
        while total_loss > 0.02:
            #ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            #ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            #UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)
            t_loss = 0
            w = 0
            for x, y in dataset.iterate_once(self.batch_size):
                # x es el vector de entrada e y es el gold               
                loss = self.get_loss(x, y)
                gradientes = nn.gradients(loss, [self.w0, self.w1, self.w2])
                
                # Actualización
                for i, wi in enumerate([self.w0, self.w1, self.w2]):
                    wi.update(gradientes[i], self.lr)
                
                w += 1
                t_loss += loss.data
                
            total_loss = t_loss / w
            print(total_loss)      
        
        # Fin de entrenamiento
        #input()




            
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # TEN ENCUENTA QUE TIENES 10 CLASES, ASI QUE LA ULTIMA CAPA TENDRA UNA SALIDA DE 10 VALORES,
        # UN VALOR POR CADA CLASE

        output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 CLASES
        pixel_dim_size = 784
        pixel_vector_length = pixel_dim_size* pixel_dim_size
 
        "*** YOUR CODE HERE ***"
        # La entrada es de 1x784 ya que hay 784 pixeles
        
        self.batch_size = 20
        self.lr = -0.01
        
        n = 256
        m = 128
        
        self.w0 = nn.Parameter(pixel_dim_size, n)
        self.b0 = nn.Parameter(1, n)
        
        self.w1 = nn.Parameter(n, m)
        self.b1 = nn.Parameter(1, m)
        
        self.w2 = nn.Parameter(m, output_size)
        self.b2 = nn.Parameter(1, output_size)
     

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
            output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 "COSENOS"
        """
        "*** YOUR CODE HERE ***"


        o1 = nn.Linear(x, self.w0)
        o1 = nn.AddBias(o1, self.b0)
        o1 = nn.ReLU( o1 )
        
        
        o2 = nn.Linear(o1, self.w1)
        o2 = nn.AddBias(o2, self.b1)
        o2 = nn.ReLU( o2 )
        
        o3 = nn.Linear(o2, self.w2)
        o3 = nn.AddBias(o3, self.b2)
        #o3 = nn.ReLU( o3 )
        
        return o3


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        POR EJEMPLO: [0,0,0,0,0,1,0,0,0,0,0] seria la y correspondiente al 5
                     [0,1,0,0,0,0,0,0,0,0,0] seria la y correspondiente al 1

        EN ESTE CASO ESTAMOS HABLANDO DE MULTICLASS, ASI QUE TIENES QUE CALCULAR 
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"#NO ES NECESARIO QUE LO IMPLEMENTEIS, SE OS DA HECHO
        return nn.SoftmaxLoss(self.run(x), y) # COMO VEIS LLAMA AL RUN PARA OBTENER POR CADA BATCH
                                              # LOS 10 VALORES DEL "COSENO". TENIENDO EL Y REAL POR CADA EJEMPLO
                                              # APLICA SOFTMAX PARA CALCULAR LA PROBABILIDA MAX
                                              # Y ESA SERA SU PREDICCION,
                                              # LA CLASE QUE MUESTRE EL MAYOR PROBABILIDAD, LA PREDICCION MAS PROBABLE, Y LUEGO LA COMPARARA CON Y 

    def train(self, dataset):
        """
        Trains the model.
        EN ESTE CASO EN VEZ DE PARAR CUANDO EL ERROR SEA MENOR QUE UN VALOR O NO HAYA ERROR (CONVERGENCIA),
        SE PUEDE HACER ALGO SIMILAR QUE ES EN NUMERO DE ACIERTOS. EL VALIDATION ACCURACY
        NO LO TENEIS QUE IMPLEMENTAR, PERO SABED QUE EMPLEA EL RESULTADO DEL SOFTMAX PARA CALCULAR
        EL NUM DE EJEMPLOS DEL TRAIN QUE SE HAN CLASIFICADO CORRECTAMENTE 
        """
        while dataset.get_validation_accuracy() < 0.97:
            #ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            #ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            #UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)
            "*** YOUR CODE HERE ***"
            for x, y in dataset.iterate_once(self.batch_size):
                # x es el vector de entrada e y es el gold               
                loss = self.get_loss(x, y)
                gradientes = nn.gradients(loss, [self.w0, self.w1, self.w2])
                
                # Actualización
                for i, wi in enumerate([self.w0, self.w1, self.w2]):
                    wi.update(gradientes[i], self.lr)
                
                i += 1  
        
        # Fin de entrenamiento
        #input()




