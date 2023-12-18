
<a name="readme-top"></a>
<div align="center">

  

  <h1>Artificial-Intelligence problema 8 regine</h1>
  
  <p>
  Il problema consiste nel disporre 8 regine all'interno di una scacchiera regolamentare 8x8, in modo che nessuna possa minacciarne o sia minacciata da un'altra. Ricordiamo che una regina può muoversi di quante caselle vuole, in orizzontale, in verticale e in diagonale. Il problema è affrontabile e risolvibile seguendo percorsi differenti. Ognuno con efficienza e prestazioni molto diverse tra loro. 
  </p>

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

<!-- Consegna -->
<details>

<summary>

# :notebook_with_decorative_cover: Testo Esercizio

</summary>


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
</details>
