This is a C++ project
I progrom this project in VS2017
If you are using Visual studio, you could click HW!.sln to open this project.

Run this project, you need put any test file in the folder"HW1", in the same path of “Search.cpp“” in order to find this test file  
OR
Put any test file in the folder"Debug", in the same path of "HW1.exe", then clik "HW1.exe" to run this project

Then run the project
accoding to the command to run this project.

output produced by my program of two samples and the bonus problem are in this folder too.
The "bonus problem.txt" is the bonus problem graph
and the output of the "bonus problem" of my problem is : Please input input file name:
graph2.txt
 ·Depth 1st search

   Expanded     Queue
      S         [<S>]
      A         [<A,S> <D,S> ]
      B         [<B,A,S> <D,A,S> <D,S> ]
      C         [<C,B,A,S> <E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      E         [<E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      D         [<D,E,B,A,S> <F,E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      F         [<F,E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      G         [<G,F,E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
goal reached!
 ·Breadth 1st search

   Expanded     Queue
      S         [<S>]
      A         [<A,S> <D,S> ]
      D         [<D,S> <B,A,S> <D,A,S> ]
      B         [<B,A,S> <D,A,S> <A,D,S> <E,D,S> ]
      D         [<D,A,S> <A,D,S> <E,D,S> <C,B,A,S> <E,B,A,S> <L,B,A,S> ]
      A         [<A,D,S> <E,D,S> <C,B,A,S> <E,B,A,S> <L,B,A,S> <E,D,A,S> ]
      E         [<E,D,S> <C,B,A,S> <E,B,A,S> <L,B,A,S> <E,D,A,S> <B,A,D,S> ]
      C         [<C,B,A,S> <E,B,A,S> <L,B,A,S> <E,D,A,S> <B,A,D,S> <B,E,D,S> <F,E,D,S> ]
      E         [<E,B,A,S> <L,B,A,S> <E,D,A,S> <B,A,D,S> <B,E,D,S> <F,E,D,S> ]
      L         [<L,B,A,S> <E,D,A,S> <B,A,D,S> <B,E,D,S> <F,E,D,S> <D,E,B,A,S> <F,E,B,A,S> ]
      E         [<E,D,A,S> <B,A,D,S> <B,E,D,S> <F,E,D,S> <D,E,B,A,S> <F,E,B,A,S> <M,L,B,A,S> ]
      B         [<B,A,D,S> <B,E,D,S> <F,E,D,S> <D,E,B,A,S> <F,E,B,A,S> <M,L,B,A,S> <B,E,D,A,S> <F,E,D,A,S> ]
      B         [<B,E,D,S> <F,E,D,S> <D,E,B,A,S> <F,E,B,A,S> <M,L,B,A,S> <B,E,D,A,S> <F,E,D,A,S> <C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> ]
      F         [<F,E,D,S> <D,E,B,A,S> <F,E,B,A,S> <M,L,B,A,S> <B,E,D,A,S> <F,E,D,A,S> <C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> ]
      D         [<D,E,B,A,S> <F,E,B,A,S> <M,L,B,A,S> <B,E,D,A,S> <F,E,D,A,S> <C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> ]
      F         [<F,E,B,A,S> <M,L,B,A,S> <B,E,D,A,S> <F,E,D,A,S> <C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> ]
      M         [<M,L,B,A,S> <B,E,D,A,S> <F,E,D,A,S> <C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> ]
      B         [<B,E,D,A,S> <F,E,D,A,S> <C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> ]
      F         [<F,E,D,A,S> <C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> ]
      C         [<C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> <G,F,E,D,A,S> ]
      E         [<E,B,A,D,S> <L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> <G,F,E,D,A,S> ]
      L         [<L,B,A,D,S> <A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> <G,F,E,D,A,S> <F,E,B,A,D,S> ]
      A         [<A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> <G,F,E,D,A,S> <F,E,B,A,D,S> <M,L,B,A,D,S> ]
      C         [<C,B,E,D,S> <L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> <G,F,E,D,A,S> <F,E,B,A,D,S> <M,L,B,A,D,S> ]
      L         [<L,B,E,D,S> <G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> <G,F,E,D,A,S> <F,E,B,A,D,S> <M,L,B,A,D,S> ]
      G         [<G,F,E,D,S> <G,F,E,B,A,S> <G,M,L,B,A,S> <C,B,E,D,A,S> <L,B,E,D,A,S> <G,F,E,D,A,S> <F,E,B,A,D,S> <M,L,B,A,D,S> <M,L,B,E,D,S> ]
goal reached!
 ·Depth-limited search (depth-limit = 2)

   Expanded     Queue
      S         [<S>]
      A         [<A,S> <D,S> ]
      B         [<B,A,S> <D,A,S> <D,S> ]
      D         [<D,A,S> <D,S> ]
      D         [<D,S> ]
      A         [<A,D,S> <E,D,S> ]
      E         [<E,D,S> ]
Search Failed!
 ·Iterative deepening search

   Expanded     Queue
L=0   S         [<S>]

L=1   S         [<S>]
      A         [<A,S> <D,S> ]
      D         [<D,S> ]

L=2   S         [<S>]
      A         [<A,S> <D,S> ]
      B         [<B,A,S> <D,A,S> <D,S> ]
      D         [<D,A,S> <D,S> ]
      D         [<D,S> ]
      A         [<A,D,S> <E,D,S> ]
      E         [<E,D,S> ]

L=3   S         [<S>]
      A         [<A,S> <D,S> ]
      B         [<B,A,S> <D,A,S> <D,S> ]
      C         [<C,B,A,S> <E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      E         [<E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      L         [<L,B,A,S> <D,A,S> <D,S> ]
      D         [<D,A,S> <D,S> ]
      E         [<E,D,A,S> <D,S> ]
      D         [<D,S> ]
      A         [<A,D,S> <E,D,S> ]
      B         [<B,A,D,S> <E,D,S> ]
      E         [<E,D,S> ]
      B         [<B,E,D,S> <F,E,D,S> ]
      F         [<F,E,D,S> ]

L=4   S         [<S>]
      A         [<A,S> <D,S> ]
      B         [<B,A,S> <D,A,S> <D,S> ]
      C         [<C,B,A,S> <E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      E         [<E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      D         [<D,E,B,A,S> <F,E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      F         [<F,E,B,A,S> <L,B,A,S> <D,A,S> <D,S> ]
      L         [<L,B,A,S> <D,A,S> <D,S> ]
      M         [<M,L,B,A,S> <D,A,S> <D,S> ]
      D         [<D,A,S> <D,S> ]
      E         [<E,D,A,S> <D,S> ]
      B         [<B,E,D,A,S> <F,E,D,A,S> <D,S> ]
      F         [<F,E,D,A,S> <D,S> ]
      D         [<D,S> ]
      A         [<A,D,S> <E,D,S> ]
      B         [<B,A,D,S> <E,D,S> ]
      C         [<C,B,A,D,S> <E,B,A,D,S> <L,B,A,D,S> <E,D,S> ]
      E         [<E,B,A,D,S> <L,B,A,D,S> <E,D,S> ]
      L         [<L,B,A,D,S> <E,D,S> ]
      E         [<E,D,S> ]
      B         [<B,E,D,S> <F,E,D,S> ]
      A         [<A,B,E,D,S> <C,B,E,D,S> <L,B,E,D,S> <F,E,D,S> ]
      C         [<C,B,E,D,S> <L,B,E,D,S> <F,E,D,S> ]
      L         [<L,B,E,D,S> <F,E,D,S> ]
      F         [<F,E,D,S> ]
      G         [<G,F,E,D,S> ]
goal reached!
 ·Uniform Search (Branch-and-bound)

   Expanded     Queue
      S         [0.0<S> ]
      A         [3.0<A,S> 4.0<D,S> ]
      D         [4.0<D,S> 7.0<B,A,S> 8.0<D,A,S> ]
      E         [6.0<E,D,S> 7.0<B,A,S> 8.0<D,A,S> 9.0<A,D,S> ]
      B         [7.0<B,A,S> 8.0<D,A,S> 9.0<A,D,S> 10.0<F,E,D,S> 11.0<B,E,D,S> ]
      D         [8.0<D,A,S> 9.0<A,D,S> 10.0<F,E,D,S> 11.0<B,E,D,S> 11.0<C,B,A,S> 11.0<L,B,A,S> 12.0<E,B,A,S> ]
      A         [9.0<A,D,S> 10.0<E,D,A,S> 10.0<F,E,D,S> 11.0<B,E,D,S> 11.0<C,B,A,S> 11.0<L,B,A,S> 12.0<E,B,A,S> ]
      E         [10.0<E,D,A,S> 10.0<F,E,D,S> 11.0<B,E,D,S> 11.0<C,B,A,S> 11.0<L,B,A,S> 12.0<E,B,A,S> 13.0<B,A,D,S> ]
      F         [10.0<F,E,D,S> 11.0<B,E,D,S> 11.0<C,B,A,S> 11.0<L,B,A,S> 12.0<E,B,A,S> 13.0<B,A,D,S> 14.0<F,E,D,A,S> 15.0<B,E,D,A,S> ]
      B         [11.0<B,E,D,S> 11.0<C,B,A,S> 11.0<L,B,A,S> 12.0<E,B,A,S> 13.0<B,A,D,S> 13.0<G,F,E,D,S> 14.0<F,E,D,A,S> 15.0<B,E,D,A,S> ]
      C         [11.0<C,B,A,S> 11.0<L,B,A,S> 12.0<E,B,A,S> 13.0<B,A,D,S> 13.0<G,F,E,D,S> 14.0<F,E,D,A,S> 15.0<A,B,E,D,S> 15.0<B,E,D,A,S> 15.0<C,B,E,D,S> 15.0<L,B,E,D,S> ]
      L         [11.0<L,B,A,S> 12.0<E,B,A,S> 13.0<B,A,D,S> 13.0<G,F,E,D,S> 14.0<F,E,D,A,S> 15.0<A,B,E,D,S> 15.0<B,E,D,A,S> 15.0<C,B,E,D,S> 15.0<L,B,E,D,S> ]
      E         [12.0<E,B,A,S> 13.0<B,A,D,S> 13.0<G,F,E,D,S> 14.0<F,E,D,A,S> 14.0<M,L,B,A,S> 15.0<A,B,E,D,S> 15.0<B,E,D,A,S> 15.0<C,B,E,D,S> 15.0<L,B,E,D,S> ]
      B         [13.0<B,A,D,S> 13.0<G,F,E,D,S> 14.0<D,E,B,A,S> 14.0<F,E,D,A,S> 14.0<M,L,B,A,S> 15.0<A,B,E,D,S> 15.0<B,E,D,A,S> 15.0<C,B,E,D,S> 15.0<L,B,E,D,S> 16.0<F,E,B,A,S> ]
      G         [13.0<G,F,E,D,S> 14.0<D,E,B,A,S> 14.0<F,E,D,A,S> 14.0<M,L,B,A,S> 15.0<A,B,E,D,S> 15.0<B,E,D,A,S> 15.0<C,B,E,D,S> 15.0<L,B,E,D,S> 16.0<F,E,B,A,S> 17.0<C,B,A,D,S> 17.0<L,B,A,D,S> 18.0<E,B,A,D,S> ]
goal reached!
 ·Greedy search

   Expanded     Queue
      S         [11.0<S> ]
      D         [8.9<D,S> 10.4<A,S> ]
      E         [6.9<E,D,S> 10.4<A,S> 10.4<A,D,S> ]
      B         [2.7<B,E,D,S> 3.0<F,E,D,S> 10.4<A,S> 10.4<A,D,S> ]
      F         [3.0<F,E,D,S> 4.0<C,B,E,D,S> 4.0<L,B,E,D,S> 10.4<A,D,S> 10.4<A,S> 10.4<A,B,E,D,S> ]
      G         [0.0<G,F,E,D,S> 4.0<C,B,E,D,S> 4.0<L,B,E,D,S> 10.4<A,D,S> 10.4<A,S> 10.4<A,B,E,D,S> ]
goal reached!
 ·A*

   Expanded     Queue
      S         [11.0<S> ]
      D         [12.9<D,S> 13.4<A,S> ]
      E         [12.9<E,D,S> 13.4<A,S> ]
      F         [13.0<F,E,D,S> 13.4<A,S> 13.7<B,E,D,S> ]
      G         [13.0<G,F,E,D,S> 13.4<A,S> 13.7<B,E,D,S> ]
goal reached!
 ·Hill-Climbing

   Expanded     Queue
      S         [11.0<S> ]
      D         [8.9<D,S> 10.4<A,S> ]
      E         [6.9<E,D,S> 10.4<A,D,S> ]
      B         [2.7<B,E,D,S> 3.0<F,E,D,S> ]
      C         [4.0<C,B,E,D,S> 4.0<L,B,E,D,S> 10.4<A,B,E,D,S> ]
Search Failed!
 ·Beam search (w = 2)

   Expanded     Queue
      S         [11.0<S> ]
      A         [10.4<A,S> 8.9<D,S> ]
      D         [8.9<D,S> 2.7<B,A,S> 8.9<D,A,S> ]
      B         [2.7<B,A,S> 6.9<E,D,S> ]
      E         [6.9<E,D,S> 4.0<C,B,A,S> 6.9<E,B,A,S> 4.0<L,B,A,S> ]
      B         [2.7<B,E,D,S> 3.0<F,E,D,S> ]
      F         [3.0<F,E,D,S> 10.4<A,B,E,D,S> 4.0<C,B,E,D,S> 4.0<L,B,E,D,S> ]
      C         [4.0<C,B,E,D,S> 0.0<G,F,E,D,S> ]
      G         [0.0<G,F,E,D,S> ]
goal reached!