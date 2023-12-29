
<a name="readme-top"></a>
<div align="center">

  

  <h1>Artificial-Intelligence </h1>
  
 

<!-- Badges -->
<p>
  <a href="https://github.com/EmmanuelloDaniele/3D-Porfolio/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/EmmanuelloDaniele/Artificial-Intelligence" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/EmmanuelloDaniele/Artificial-Intelligence" alt="last update" />
  </a>
  <a href="https://github.com/EmmanuelloDaniele/3D-Porfolio/network/members">
    <img src="https://img.shields.io/github/forks/EmmanuelloDaniele/3Artificial-Intelligence" alt="forks" />
  </a>
  <a href="https://github.com/EmmanuelloDaniele/3D-Porfolio/stargazers">
    <img src="https://img.shields.io/github/stars/EmmanuelloDaniele/Artificial-Intelligence" alt="stars" />
  </a>
  <a href="https://github.com/EmmanuelloDaniele/3D-Porfolio/issues/">
    <img src="https://img.shields.io/github/issues/EmmanuelloDaniele/Artificial-Intelligence" alt="open issues" />
  </a>
  <a href="https://github.com/EmmanuelloDaniele/3D-Porfolio/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/EmmanuelloDaniele/Threejs_3D_Portfolio.svg" alt="license" />
  </a>
</p>
   
 <h4>
  <span> · </span>
    <a href="https://github.com/EmmanuelloDaniele/Artificial-Intelligence/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/EmmanuelloDaniele/Artificial-Intelligence/issues/">Request Feature</a>
  </h4>
</div>

<br />
<!-- Deep Learning Image Classifier -->
<details>

<summary>

# :notebook_with_decorative_cover: Deep Learning Image Classification


</summary>
# SGD: Stochastic Gradient Descent
It is an optimization algorithm used to train neural networks and minimize the cost function, aiming to find the global minimum. The stochastic approach stems from the fact that the gradient is calculated on random subsets of the training data rather than the entire dataset.




<h2>1. Install Dependencies and Setup</h2>
In [ ]: <code>!pip install tensorflow tensorflow-gpu opencv-python matplotlib</code></br>
In [ ]: <code>!pip list</code></br>
In [1]: <code>import tensorflow as tf</code></br>
In [1]: <code>import os</code></br>
 # Avoid OOM errors by setting GPU Memory Consumption Growth</br>
In [2]: <code>gpus = tf.config.experimental.list_physical_devices('GPU')</code></br>
In [2]: <code>for gpu in gpus:</code></br>
In [2]: <code>tf.config.experimental.set_memory_growth(gpu, True)</code></br>
In [3]: <code>tf.config.list_physical_devices('GPU')</code></br>

Out [3]: <code>[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')] </code></br>

<h2>2. Remove dodgy images</h2>
In [4]: <code>import cv2</code></br>
In [4]: <code>import imghdr</code></br>
In [6]: <code>data_dir = 'data' </code></br>
In [6]: <code>image_exts = ['jpeg','jpg', 'bmp', 'png']</code></br>
 </br>
In [7]: <code>for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)</code></br>

<h2>3. Load Data</h2>
In [8]: <code>import numpy as np
from matplotlib import pyplot as plt</code></br>
In [9]: <code>data = tf.keras.utils.image_dataset_from_directory('data')
</code></br>
Out [9]: Found 305 files belonging to 2 classes.</br>
In [10]: <code>data_iterator = data.as_numpy_iterator()
</code></br>
In [11]: <code>batch = data_iterator.next()</code></br>
In [12]: <code>for image_class in os.listdir(data_dir): 
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])</code></br>

<h2>4. Scale Data</h2>
In [13]: <code>data = data.map(lambda x,y: (x/255, y))</code></br>
In [ ]: <code>data.as_numpy_iterator().next()</code></br>


<h2>5. Split Data</h2>
In [15]: <code>train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)</code></br>
In [16]: <code>train_size</code></br>
Out [16]: 7</br>
In [17]: <code>train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
</code></br>

<h2>6. Build Deep Learning Model
</h2>
In [18]: <code>train</code></br>
Out [18]: <code><TakeDataset element_spec=(TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>
</code></br>
In [19]: <code>from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout</code></br>
In [20]: <code>model = Sequential()
</code></br>
In [21]: model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))</br>
</code></br>
In [22]: <code>model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
</code></br>
In [23]: <code>model.summary()
</code></br>
Out [23]: <code>Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 254, 254, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 127, 127, 16)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 125, 125, 32)      4640      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 62, 62, 32)       0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 60, 60, 16)        4624      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 30, 30, 16)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 14400)             0         
                                                                 
 dense (Dense)               (None, 256)               3686656   
                                                                 
 dense_1 (Dense)             (None, 1)                 257       
                                                                 
=================================================================
Total params: 3,696,625
Trainable params: 3,696,625
Non-trainable params: 0
_________________________________________________________________
</code></br>

In [22]: <code>
</code></br>

<h2>7. Train
</h2>
In [24]: <code>logdir='logs'

</code></br>
In [25]: <code>tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

</code></br>
In []: <code>hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

</code></br>

<h2>8. Plot Performance Loss, Accuracy
</h2>
In [27]: <code>fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
</code></br>
In [28]: <code>fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()</code></br>

<h2>9. Evaluate
</h2>
In [29]: <code>from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
</code></br>
In [30]: <code>pre = Precision()
re = Recall()
acc = BinaryAccuracy()
</code></br>
In [31]: <code> for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)
    </code></br>
In [32]: <code>print(pre.result(), re.result(), acc.result())
</code></br>
Out [ ]: tf.Tensor(1.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32) tf.Tensor(1.0, shape=(), dtype=float32)</br>

<h2>10. Test
</h2>
In [33]: <code>import cv2
</code></br>
In [39]: <code>img = cv2.imread('154006829.jpg')
plt.imshow(img)
plt.show()
</code></br>
In [40]: <code> resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
    </code></br>
In [41]: <code>yhat = model.predict(np.expand_dims(resize/255, 0))

</code></br>
In [42]: <code>yhat</code></br>
Out [42]: array([[0.01972741]], dtype=float32)</br>
In [42]: <code>if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')</code></br>
  Predicted class is Happy</br>

<h2>11. Save the Model

</h2>
In [44]: <code>from tensorflow.keras.models import load_model

</code></br>
In [45]: <code>model.save(os.path.join('models','imageclassifier.h5'))

</code></br>
In [46]: <code>new_model = load_model('imageclassifier.h5')
</code></br>
In [47]: <code>new_model.predict(np.expand_dims(resize/255, 0))
</code></br>
Out [47]: array([[0.01972741]], dtype=float32)</br>

  Predicted class is Happy</br>

</details>

<!-- Rete Neurale SGD -->
<details>

<summary>

# :notebook_with_decorative_cover: Train a Neural Network(SGD)


</summary>
# SGD: Stochastic Gradient Descent
It is an optimization algorithm used to train neural networks and minimize the cost function, aiming to find the global minimum. The stochastic approach stems from the fact that the gradient is calculated on random subsets of the training data rather than the entire dataset.

<h2>To train a neural network means finding the weight values that minimize an error function on the training set.</h2>
▶ To find the optimal weights, it is necessary to use an optimization algorithm.</br>
▶ The simplest optimization algorithm is gradient descent.</br>
▶ Gradient descent is an iterative algorithm that can be applied to any differentiable function.</br>
<p>
x1\
   \
    w3
     \      W^t x
x2--w2->P------------->g(·)------------>
     /                        y
    w1
   /
  /
1/
</p>
Il neurone riceve un insieme di input x = (x1, x2, . . . , xn, 1),
ogni input xi è moltiplicato per un peso Wi.</br>
▶ I pesi W sono i parametri della rete neurale.</br>
▶ I pesi W sono inizializzati casualmente.</br>
▶ I pesi W sono aggiornati durante l’addestramento</br>
Consideriamo una rete neurale con un solo neurone.
La rete deve imparare a separare due classi linearmente separabili sul piano
e ha come funzione di attivazione la funzione identità.</br>
g(x) = x</br>
g′(x) = 1</br>
Dataset:</br>
 x1 |  x2  | y</br>
2.0 |  1.0 | 1</br>
6.0 |  0.5 |-1</br>
2.5 | -1.0 | 1</br>
5.0 |  0.0 |-1</br>
0.0 |  0.0 | 1</br>
4.0 | -1.0 |-1</br>
1.0 |  0.5 | 1</br>
3.0 |  1.5 |-1</br>
</details>
<!-- 8 Regine -->
<details>

<summary>

# :notebook_with_decorative_cover: 8 Queens Problem

</summary>

 <p>
  Il problema consiste nel disporre 8 regine all'interno di una scacchiera regolamentare 8x8, in modo che nessuna possa minacciarne o sia minacciata da un'altra. Ricordiamo che una regina può muoversi di quante caselle vuole, in orizzontale, in verticale e in diagonale. Il problema è affrontabile e risolvibile seguendo percorsi differenti. Ognuno con efficienza e prestazioni molto diverse tra loro. 
  </p>
<h1>Testo esercizio</h1>
Si scriva un programma Python per determinare le soluzioni del rompicapo delle otto regine. Il rompicapo
delle otto regine è un problema che consiste nel trovare il modo di posizionare otto regine (pezzo degli
scacchi) su una scacchiera 8 × 8 con una disposizione tale che nessuna di esse possa catturarne un’altra,
usando i movimenti standard della regina. Perciò, una soluzione dovrà prevedere che nessuna regina
abbia una colonna, riga o diagonale in comune con un’altra regina.
Si codifichi lo lo stato del problema come una lista, dove ogni elemento della lista rappresenta la colonna in
cui è posizionata la regina della riga corrispondente. La scacchiera in figura sarebbe quindi codificata come
[6, 2, 7, 1, 4, 0, 5, 3]. Lo stato iniziale sarà una lista vuota, e ogni azione consiste nell’aggiungere
una regina nella riga successiva.
Suggerimento: Si definisca una funzione is_valid(state) che, dato uno stato, restituisce True se lo
stato è valido, False se contiene due regine sulla stessa colonna o sulla stessa diagonale (la codifica dello
stato impedisce che due regine siano sulla stessa riga).

<h2>Approccio intuitivo iniziale</h2>

Partiamo dalla soluzione "banale". 

La soluzione consiste nel generare tutte le disposizioni possibili di 8 regine su una scacchiera di  8x8 == 64 caselle. Dal Coefficiente binomiale, calcoliamo che queste disposizioni ammontano a 4.426.165.368. Scriviamo un programma che genera tutte le disposizioni, e scartiamo tutte quelle dove c'è almeno una regina che ne minaccia un'altra. Le disposizioni da analizzare sono un numero decisamente elevato. Per velocizzare la soluzione, abbiamo bisogno di semplificare il problema, per esempio riducendo il numero di disposizioni da analizzare. 

<h2>Semplifica.</h2>

Una prima semplificazione potrebbe essere, per esempio, imporre una sola regina per ogni riga. Due regine nella stessa riga si minacciano in orizzontale. 

E nelle otto righe, tutte le regine stanno sicuramente disposte in posizioni (colonne) diverse. Per non minacciarsi in verticale. 

Quindi in tutta la scacchiera, nella nostra soluzione avremo sicuramente una riga con una regina nella prima posizione a sinistra, una riga con la regina in seconda posizione eccetera. 

Questo algoritmo semplificato, invece di generare tutte le disposizioni descritte sopra, prevede di generare solo le combinazioni possibili di 8 righe, ognuna con una regina in posizione diversa. Nella prima riga ho 8 posizioni possibili. Nella seconda ne ho sette (in realtà ne ho 5 oppure sei, considerando le caselle minacciate in diagonale). Nella terza ne ho sei, eccetera. Queste combinazioni sono 8 fattoriale (8! == 40320), un numero di 5 ordini di grandezza inferiore alle disposizioni calcolate sopra. Anche qui dobbiamo generare le combinazioni e scartare tutte quelle dove è presente almeno una regina che ne minaccia un'altra. 

La minaccia potrà arrivare solo in diagonale, in quanto le minacce in orizzontale e in verticale le abbiamo già eliminate “per costruzione”.
Possiamo ottimizzare ancora?

Per un problema di ordine 8, circa quarantamila combinazioni da analizzare sono relativamente poche. Ma cosa succede se vogliamo risolvere il problema con 16 regine su una scacchiera 16x16? Oppure se volessimo risolvere il problema con 24? 24 Fattoriale (24!) è un numero dell’ordine di 10^23. Se avessimo un miliardo di calcolatori, ognuno in grado di esaminare un miliardo di combinazioni al secondo, potremmo calcolare tutte le soluzioni in circa una settimana di calcolo.  Decisamente troppe combinazioni da generare e da esaminare in tempi “umani” con risorse limitate. Abbiamo bisogno di semplificare ancora il problema, perchè questo sia affrontabile. 

<h2>Ma torniamo al problema di ordine 8. </h2>

Per esempio, possiamo pensare di cominciare a controllare le minacce in diagonale anche con le combinazioni incomplete, mentre le costruiamo, senza aspettare di aver piazzato tutte le regine per farlo. Per ogni riga, quindi, prima di piazzare una regina nella casella candidata, controllo le minacce in diagonale su questa casella. Se la casella è minacciata, esamino la successiva nella riga. Dispongo la regina nella posizione non minacciata e non ancora provata in precedenza. Se arrivo alla fine della riga senza altre posizioni disponibili, torno alla riga precedente, dove applico la stessa logica, cercando una nuova disposizione per quella regina. Quando arrivo a disporre tutte le 8 regine sulle 8 righe, la soluzione è valida, la conteggio e la stampo o la memorizzo. E continuo la ricerca. Quando ho generato tutte le disposizioni possibili, mi fermo. 

Questo approccio si presta molto bene ad un'implementazione ricorsiva. Se nella riga attuale non sono rimaste altre posizioni utilizzabili, perchè tutte già provate o minacciate dalle regine disposte nelle righe precedenti, interrompo la ricerca sulla riga attuale e innesco il backtracking tornando alla riga precedente. Non provando (e quindi non verificando) tutte le combinazioni complete possibili, è chiaramente molto più efficiente dei precedenti. Perchè quando innesco il backtracking per mancanza di posizioni sulla riga attuale, sto in realtà scartando una grossa porzione di combinazioni da analizzare. Sto scartando in un colpo solo tutto il sottoalbero di combinazioni possibili, ma sicuramente non valide. Albero che può essere decisamente grande. 
<h2>Come capiamo se una casella è minacciata?</h2>

Come già accennato, se vogliamo estendere il problema da 8 a 12, 16 o addirittura 24 regine, rispettivamente su scacchiere 12x12, 16x16 o 24x24, le prime soluzioni proposte sopra si scontrano contro la complessità esponenziale e fattoriale del problema. E diventano quindi inapplicabili. Il metodo ricorsivo rimane invece ancora applicabile, anche per dimensioni del problema più grandi di 8. Ora ci serve un metodo veloce per capire se la casella analizzata è minacciata dalle regine già disposte sopra. Accoppiato con un metodo efficiente per rappresentare questi dati in memoria, in modo da poter fare sia le disposizioni delle regine che le verifiche sulle minacce molto velocemente. Intuitivamente si potrebbe usare una matrice 8x8 per rappresentare la scacchiera. Come vedremo in seguito, esistono strutture più compatte ed efficienti, data la natura dei dati e dei calcoli da fare. Per ora usiamo una matrice per rappresentare graficamente quello che vogliamo fare. Usiamo un simbolo 'R' per piazzare una regina in una riga della matrice, e un simbolo 'M' per dire che una casella è minacciata.

Numeriamo righe e colonne a partire dall'angolo in alto a sinistra, con i numeri di riga e colonna che vanno da 0 a 7. All’inizio, piazziamo la prima regina in (0,0) (riga,colonna) e segnamo le caselle minacciate. 

Per piazzare la seconda regina nella seconda riga usiamo la casella nella terza colonna, in (1,2). Perchè (1,1) è minacciata in diagonale. E anche qui marchiamo le caselle minacciate. 

Per piazzare la terza regina nella terza riga (riga 2), possiamo piazzarla solo a partire dalla quinta colonna (2,4), in quanto le precedenti da (2,0) a (2,3) sono tutte minacciate. 

E così via. Come si vede, ogni volta che piazziamo una regina si evitano diverse combinazioni da esplorare in seguito, per via delle caselle minacciate. Nella riga 3 per esempio cominciamo ad esplorare l'albero con la regina in (3,1) per vedere se contiene soluzioni. Finito di esplorare questo, proveremo con la regina in (3,6), scartando le precedenti da (3,2) a (3,5), in quanto minacciate. Nella riga 5 abbiamo solo la possibilità di piazzare la nostra regina in (5,3).

<h2>E se la minaccia arriva in diagonale?</h2>

Se come accennato sopra, per costruzione piazziamo una regina per riga, in una posizione diversa dalle regine delle righe precedenti, escludiamo subito le minacce in orizzontale e in verticale. Dobbiamo "solo" verificare se la posizione “candidata” è minacciata in diagonale dalle regine piazzate in precedenza. Per esempio, possiamo dire che una regina in posizione (4,5), è minacciata dalla regina nella riga precedente (riga 3) solo se questa regina si trova in posizione (3,4) o in posizione (3,6). Le caselle in diagonale in alto a destra e a sinistra di (4,5).

Se lo è, ci fermiamo e troviamo la prossima casella candidata. Se non lo è, andiamo e controlliamo se è minacciata dalla regina in riga 2. La casella (4,5) è minacciata dalla regina in riga 2 se questa si trova in posizione (2,3) o in posizione (2,7). 

E così via, fino a controllare se la nostra casella candidata (4,5) è minacciata dalla regina nella  prima riga (riga 0). Dobbiamo quindi generare le coordinate delle caselle nelle due direzioni diagonali in alto a destra e in alto a sinistra della posizione che stiamo controllando, e verificare se a queste coordinate è presente una regina. 

Per (4,5), che è la nostra casella candidata, queste caselle “minaccianti” sono: 

(3,4), (2,3), (1,2), (0,1) per la diagonale a sinistra 

(3,6), (2,7) per la diagonale destra 

Generalizzando, per verificare se la casella (j,k) è minacciata in diagonale dalle regine disposte sopra, dobbiamo verificare, riga per riga a ritroso partendo dalla riga j, la presenza di una regina nelle caselle che hanno colonna k-1 o k+1 nella riga j-1, k-2 o k+2 nella riga j-2, k-3 o k+3 nella riga j-3 eccetera, indietro fino alla riga 0. Se otteniamo una colonna negativa (minore di zero) o maggiore di 7, ovviamente possiamo ignorare il check. 

</details>





<!-- Contact -->


##  More info  :handshake: Contact

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/emmanuello) 
<p dir="auto">Daniele Emmanuello - <a href="https://www.linkedin.com/in/emmanuellodaniele/" rel="nofollow">@Linkedin</a> -<a href="https://t.me/emmanuellodaniele"rel="nofollow">@Telegram</a></p> 

<p align="right">(<a href="#readme-top">back to top</a>)</p>
