<TeXmacs|2.1.1>

<style|ieeeconf>

<\body>
  <doc-data|<\doc-title>
    CS303 Artificial Inteligence Project Report:\ 

    Capacitated Arc Routing Problems(CARP)
  </doc-title>|<doc-author|<author-data|<author-name|Zitong
  Huang>|<\author-affiliation>
    12012710@mail.sustech.edu.cn

    <date|>
  </author-affiliation>>>>

  <section|Introduction>

  <subsection|Arc Routing Problem>

  Arc Routing Problems(ARP) are a category of general routing
  peoblem(GRP)<cite|1>. Different from another kind of GRP called \ Node
  Routing Problems(NRP), ARP's main serve object is edges in graph.

  <subsection|Application>

  Arc Routing Problem is widly used in daily life:

  <\quote-env>
    Arc routing problems can be applied to<nbsp><hlink|garbage
    collection|https://en.wikipedia.org/wiki/Waste_collection>,<nbsp><hlink|school
    bus|https://en.wikipedia.org/wiki/School_bus><nbsp>route planning,
    package and newspaper delivery,<nbsp><hlink|deicing|https://en.wikipedia.org/wiki/Deicing><nbsp>and<nbsp><hlink|snow
    removal|https://en.wikipedia.org/wiki/Snow_removal><nbsp>with<nbsp><hlink|winter
    service vehicles|https://en.wikipedia.org/wiki/Winter_service_vehicle><nbsp>that
    sprinkle<nbsp><hlink|salt|https://en.wikipedia.org/wiki/Salt><nbsp>on the
    road,<nbsp><hlink|mail delivery|https://en.wikipedia.org/wiki/Mail>,
    network maintenance,<nbsp><hlink|street
    sweeping|https://en.wikipedia.org/wiki/Street_sweeper>, police and
    security guard patrolling, and<nbsp><hlink|snow
    ploughing|https://en.wikipedia.org/wiki/Snowplough>. Arc routings
    problems are<nbsp><hlink|NP hard|https://en.wikipedia.org/wiki/NP-hardness>,
    as opposed to<nbsp><hlink|route inspection
    problems|https://en.wikipedia.org/wiki/Route_inspection_problem><nbsp>that
    can be solved in<nbsp><hlink|polynomial-time|https://en.wikipedia.org/wiki/Polynomial-time>.<cite|1>
  </quote-env>

  In these project, a sub-problem of ARP called CARP is mainly discussed. A
  further explaination of CARP will be show later.

  <subsection|Purpose>

  This report is mainly write to explain the secand project of CS303 course.
  And, this report will give a better explanation of some algorithm used,
  theory, fomuler and method used in project code.

  <section|Preliminary Problem Description>

  <subsection|Overview of CARP>

  The CARP(capacitated arc routing problem) is a typical form of the arc
  routing problem. It is a combinatorial optimization problem with some
  constrains needed<cite|2>.

  <subsection|Describe>

  In natural language, CARP can be describe as follows:

  Given a graph with serial edges to be served. A velhicle group get task to
  serve these edges. Each velhicle have the <strong|same> capacitate. At the
  processing of serving edges, if one velhicle's total served amount reach
  its capacity, it should return to a given vertice (called depot) for at
  least one time, so that it can serve other edges continuely. Each edge have
  a cost, represent the time a velhicle would cost to travel through. \ Each
  edge need to be served have a demand, represent the demand a velhicle will
  get after serving it. CARP is to get some routes to minimize the total cost
  of this velgicle group.

  In mathmetic, CARP can be describe as follows:\ 

  <\quote-env>
    <text-dots>consider an undirected connected graph \<#1D43A\> =
    (\<#1D449\>, \<#1D438\>), with a vertex set \<#1D449\> and an edge set
    \<#1D438\> and a set of required edges (tasks) \<#1D447\> \<subseteq\>
    \<#1D438\>. A fleet of identical vehicles, each of capacity \<#1D444\>,
    is based at a designated depot vertex <math|v<rsub|0>> \<in\> \<#1D449\>.
    Each edge \<#1D452\> \<in\> \<#1D438\> incurs a cost \<#1D450\>\<#1D450\>
    \<#1D452\>\<#1D452\> whenever a vehicle travels over it or serves it (if
    it is a task). Each required edge (task) \<#1D70F\> \<in\> \<#1D447\> has
    a demand \<#1D451\>(\<#1D70F\>) \<gtr\> 0 associated with it.<cite|2>
  </quote-env>

  CARP is proved as a NP-hard Problem in \ 1981 by B. L. Golden and R. T.
  Wong<cite|3>, which means it cannot be solved in polynomial time.

  <subsection|Formulation>

  Variables use in this report is defined below:

  <\itemize>
    <item><math|G<around*|(|V,E|)>:>undirected graph with vertices set
    <math|V> and edge set <math|E>

    <item><math|s:>start vertice of the graph.

    <item><math|t>: end vertice of the graph.

    <item><math|d:>depot vertice of the graph.

    <item><math|W<rsub|i j>:>least cost when a velhical travel from vertice
    <math|i> to vertice <math|j>.\ 

    <item><math|E<rsub|d>:>edges need to be served by vehical group

    <item><math|C>:max serve a velhical can offer one time.

    <item><math|<math-it|Cost>>(<math|G,d>,<math|C>):total cost a velhical
    take after served all demand edges.\ 
  </itemize>

  Then, a carp problem can be defined as a optimized problem\VTarget function
  is <em|Cost>(<em|G,d,C>), the objective of an agent is to minimize
  <em|Cost>(<em|G,d,C>)

  \;

  Other specific name need to be assert is show below.

  <\itemize>
    <item><em|graph>: the graph in CARP, contains some edges needed to be
    served. <em|graph>[<em|i, j>] represente the cost when vehicle travel
    from vertices[i] to vertices[j].If i and j is not connected directly,
    <em|graph>[<em|i, j>] will be set as <math|\<infty\>>.

    <item><math|d[i,j]>: Shortest distance between i-th vertice and j-th
    vertice.

    <item><em|distance>: the graph after doing Floyd.\ 

    <\equation*>
      <em|<math-it|distance>>[<em|i,j>] =d[<em|i, j>]
    </equation*>

    <item><em|middle_vertice>: vertice choosed as \Pjump board\Q in Floyd
    algorithm

    <item><em|connect>ed: if vertice i and vertice j have direct path, or,
    distance[i, j] \<less\> <math|\<infty\>>

    <item><math|V>: the number of vertices in graph

    <item><math|V<rsub|i>>: the i-th vertices

    <item><math|E>: the number of edges in graph.\ 

    <item><math|E<rsub|i>>: an edge in graph.

    <item><math|E<rsub|i j>>: an edge between <math|V<rsub|i>> and
    <math|V<rsub|j>>

    <item><math|S<rsub|<math-it|edge>>>: start vertice of an edge

    <item><math|E<rsub|<math-it|edge>>>: end vertices of an edge

    <item><em|route>: element in routes, represent edges between two passes
    through the depot of a vehicle.

    <item><em|total_routes> : a result of a CARP. Contains some route.

    <item><em|demand edges>: edges consider need to be served in algorithm

    <item><em|served amount>: when a velhicle is at one moment when do
    serving, the total serve amount it served.

    <item><em|capacity>: the max served amount of one velhicle. It is always
    true that

    <\equation*>
      <math-it|served_amount>\<leqslant\><math-it|capacity>
    </equation*>

    <item><em|time>: total time procedure cost

    <item>terminate_time: max time procudure can cost. It is always true that
    time <math|\<leqslant\>> terminate time
  </itemize>

  <subsection|Solution-mark format>

  A solution of a CARP is marked as <em|routes>. A <em|routes> contains
  serial <em|route>. Each <em|route> marked as a list

  <\equation*>
    r o u t e=<around*|[|0,E<rsub|1>,E<rsub|2>\<ldots\>E<rsub|i>,0|]>
  </equation*>

  means a vehicle travels from the depot, pass through
  <math|E<rsub|1>,E<rsub|2>\<ldots\>E<rsub|i>>, then back to the depot.

  Thus, <em|total_routes> can be marked as\ 

  <\equation*>
    <math-it|total_routes>=<around*|[|<math-it|route><rsub|1>,<math-it|route><rsub|2>,<math-it|route><rsub|3>\<ldots\><math-it|route><rsub|i>|]>
  </equation*>

  <section|Procedure>

  <subsection|Introduction>

  <space|1em>Solve this problem need such procedure: data reading, distance
  calculate and routes decide. When dealing with routes decide, a
  path-scanning algorithm optimized by multi-processing and merge-split is
  used.

  <subsection|Data Reading>

  As the give document described, online jugement parse parameters for 2
  parts.\ 

  <subsubsection|First Part>

  At the first part, terminal parse basic imformation to procedure. The
  format of first parameter are shown below:

  <\itemize>
    <item>1st line: NAME : \<less\>string\<gtr\><space|1em>\U the name of
    instance

    <item>2nd line: VERTICES: \<less\>number\<gtr\><space|1em>\U number of
    vertices

    <item>3rd line: DEPOT:\<less\>number\<gtr\><space|1em>\U the depot vertex

    <item>4th line: REQUIRED EDGES:\<less\>number\<gtr\><space|1em>\U number
    of required edges

    <item>5th line: NON-REQUIRED EDGES : \<less\>number\<gtr\> \ \ \ \U the
    number of non-required edges

    <item>6th line: VEHICLES : \<less\>number\<gtr\><space|2em>\U the number
    of vehicles

    <item>7th line: CAPACITY : \<less\>number\<gtr\><space|1em>\Uthe vehicle
    capacity

    <item>8th line: TOTAL COST OF REQUIRED EDGES:
    \<less\>number\<gtr\><space|1em>\U the total cost of all tasks
  </itemize>

  <space|1em>A sample input is followed:

  <code|NAME : gdb1<next-line>VERTICES : 12<next-line>DEPOT :
  1<next-line>REQUIRED EDGES : 22<next-line>NON-REQUIRED EDGES : 0>

  <code|VEHICLES : 5<next-line>CAPACITY : 5<next-line>TOTAL COST OF REQUIRED
  EDGES : 252>

  While processing data, it is easy to found velhicle amount will not affect
  our result. Proof is not list here since it's not hard by
  <itemize|commutative law of multiplication>.

  <\subsubsection>
    Secand Part
  </subsubsection>

  In secand part, imformation of graph is parsing to the procidure.

  Input data is constitute by serial rows, each row represent an edge in
  graph. One row is divided to 4 columns, each column represent \ start_point
  of the edge, \ end_point of the edge, demand, cost, respectively. A sample
  input is show below:

  <\code>
    NODES \ \ \ \ \ \ COST \ \ \ \ \ \ \ \ DEMAND<next-line>1 \ \ 2
    \ \ \ \ \ \ \ 13 \ \ \ \ \ \ \ \ \ \ \ \ 1<next-line>1 \ \ 4
    \ \ \ \ \ \ \ 17 \ \ \ \ \ \ \ \ \ \ \ \ 1

    <text-dots>

    10 \ 11 \ \ \ \ \ \ 11 \ \ \ \ \ \ \ \ \ \ \ \ 1

    END
  </code>

  <subsection|Distance Calculate: Shortest Path Algorithm\VFloyd>

  <subsubsection|Introduce>

  When doing CARP, a shortest path algorithm is needed to compute distance
  between two vertice. Algorithm used here is Floyd algorithm.

  Main procedure of Floyd can be represented by upon:

  <\enumerate>
    <item>Initial the graph, set <em|d>[<em|i, j>] as the shortest distance
    between vertice i and vertice j. If there is no path bewteen i and j, set
    <em|d>[<em|i, j>] be <math|\<infty\>>

    <item>choose a jump-vertice k

    <item>For jump-vertice k, choose two different vertices i, j from graph.
    if <em|d>[<em|i, k>] + <em|d>[<em|k, j>] less than <em|d>[<em|i, j>], set
    <em|d>[<em|i, j>] be <em|d>[<em|i, k>]+ <em|d>[<em|k, j>]

    <item>repeat step 3 until all vertices in graph have been choosen as i or
    j at least once.(except vertice k)

    <item>repeat step 2,3 until all vertices in graph has been choosen as \ k
    as least once.
  </enumerate>

  <space|1em>The time conplexy is\ 

  <\equation*>
    O<around*|(|V<rsup|3>|)>
  </equation*>

  \;

  <space|1em>while <math|V> is the number of vertices

  <subsubsection|Formulation>

  Floyd can be expree as following:

  <\equation*>
    <math-it|distance><around*|[|i,j|]>=<math-it|Min><around*|(|<math-it|distance><around*|[|i,k|]>+<math-it|distance><around*|[|k,j|]>|)>
  </equation*>

  where k is vertice connected to both i and j.

  <subsubsection|Pseudo-code>

  \ 

  Pseudo-code is list below:

  <\render-code>
    function floyd(graph)

    begin

    <space|2em>for i=<math|V<rsub|0>> -\<gtr\><math| V<rsub|i>>

    <space|2em>begin

    <space|2em><space|2em>for j=<math|V<rsub|0>> -\<gtr\> <math| V<rsub|i>>

    <space|2em><space|2em>begin

    <space|2em><space|2em><space|2em>for k=<math|V<rsub|0>> -\<gtr\> <math|
    V<rsub|i>>

    <space|2em><space|2em><space|2em>begin

    <space|2em><space|2em><space|2em><space|2em>if graph[i, j] \<gtr\>
    graph[i, k] + graph[k, j]

    <space|2em><space|2em><space|2em><space|2em>then graph[i,j]= graph[i,k] +
    graph[k,j]

    <space|2em><space|2em><space|2em>end

    <space|2em><space|2em>end

    <space|2em>end

    <space|2em>return graph

    end
  </render-code>

  <subsection|Routes Decide>

  <subsubsection|Path-Scanning>

  <paragraph|Introduction>

  In Yao's work, a Path-Scanning algorithm is used to when \ initial the
  population while doing memetic algorithm. The core concept of Path-Scanning
  is greedy. For each edge Path-Scanning served, algorithm choose the nearest
  k edges as candinates. For elements in candinates, algorithm choose
  next-serve edge by following rules:

  <\enumerate-numeric>
    <item>maximize the distance from the head of task to the depot

    <item>minimize the distance from the head of task to the depot

    <item>maximize the term <math|dem(t)/sc(t)>, where <math|dem(t) >and
    <math|sc(t)> are demand and serving cost of task <math|t>, respectively;

    <item>minimize the term <math|dem(t)/sc(t)>

    <item>use rule 1 if the vehicle is less than half\Bfull, otherwise use
    rule 2
  </enumerate-numeric>

  As a greedy algorithm, each round of Path-Scanning has a time complexy:

  <\equation*>
    O<around*|(|E<rsup|2>|)>,while m is number of edges.
  </equation*>

  <space|2em>In this project, since merge-split operator(going to introduced
  next) failed to be used in final code, 3-rd and 4-nd is not used in order
  to keep the randomness.

  <paragraph|Formulation>

  Path-Scanning can be\ 

  \;

  <paragraph|Pseudo-code>

  Pseudo-code in this project is show following:

  <\render-code>
    function path_scanning(<em|demand_edges>)

    <space|2em><em|routes> = []

    <space|2em>while <em|demand_edge> not null

    <space|2em>begin

    <space|2em><space|2em><em|route> = []

    <space|2em><space|2em><em|last_vertice> <math|\<leftarrow\>> depot

    <space|2em><space|2em>while <em|<math|<math-it|served_amount>\<leqslant\>>capacity>

    <space|2em><space|2em><em|next_edges> <math|\<leftarrow\>>
    <strong|nearest> edge from last_vertices
    <space|1em><space|2em><space|2em><space|2em><space|2em><space|2em>in
    <em|demand_edges>

    <space|2em><space|2em>begin

    <space|2em><space|2em><space|2em>if <em|served_amount>
    <math|\<leqslant\>><em|> <em|capacity> / 2

    <space|2em><space|2em><space|2em>begin

    <space|2em><space|2em><space|2em><space|2em><em|next_edge>
    <math|\<leftarrow\>> <strong|furthest> edge from depot <space|14em>in
    <em|next_edges>.

    <space|2em><space|2em><space|2em>end

    <space|2em><space|2em><space|2em>else if <em|served_amount>
    <math|\<gtr\>> <em|capacity> / 2

    <space|2em><space|2em><space|2em>begin

    <space|2em><space|2em><space|2em><space|2em><em|next_edge>
    <math|\<leftarrow\>> <strong|nearest> edge from depot <space|14em>in
    <em|next_edges>.

    <space|2em><space|2em><space|2em>end

    <space|2em><space|2em>end

    <space|2em><space|2em><em|route> <math|\<leftarrow\>> <em|route +
    next_edge>

    <space|2em><space|2em>remove <em|nect_edge> from
    <em|demand_<em|<em|edge>>>

    <space|2em><space|2em><em|last_vertices> <math|\<leftarrow\>>
    <em|E<math|<rsub|next_edge>>><math|<rsub|>>

    <space|2em><space|2em><em|routes> = <em|routes + route>

    <space|2em>end

    return <em|routes>
  </render-code>

  <subsubsection|Merge-Split Operator>

  Merge-Split Operator is first intruduced by X.Yao in 2009<cite|2>, by using
  it as memetic algorithm's local-search operator. With Merge-Split used, Yao
  had improved the performance of memetic algorithm to a new level.

  A basic procedure of Merge-split is following:

  <\enumerate>
    <item>Initial <math|r> by using path-scanning. The route is not neccesary
    to be the best result.

    <item>Choose k routes randomly, save edges in these rooutes.

    <item>Do path-scanning again. Append result return to remained routes.
  </enumerate>

  function merge-split(route)

  begin

  \ <space|2em>split_point = random(1, length(route))

  \ <space|2em>subroute1 = route[1...split_point]

  \ <space|2em>subroute2 = route[split_point+1...length(route)]

  <space|2em>merge_distance = distance(subroute1[end], sub-

  <space|13em>route2[start])

  \ <space|2em>merged_route = subroute1 +merge_distance + sub-
  <space|10em>route2

  <space|2em>return merged_route

  end

  \;

  <subsection|Working \]ow>

  <\enumerate>
    <item>Data reading

    <item>Calculate distances between each vertices. Store the information is
    distance.

    <item>while time \<less\> terminate_time / 2, \ do path_scanning to get
    result routes

    <item>while time \<gtr\> terminate_time / 2, do merge-split to updat
    routes exist.

    <item>In result routes, choose least cost routes.
  </enumerate>

  <section|Experiments>

  <subsection|Environment>

  <subsubsection|Hardware Environment>

  CPU: Intel(R) Core(TM) i7-10875H CPU @ 2.30GHz

  GPU: NVIDIA GeForce RTX 2060

  <subsubsection|Software Environment>

  Operation System: Ubuntu 20.04 LTS

  Python: Python 3.9.12

  IDE: Visual Studio Code 1.72.2 with python extension

  Numpy: 1.16.1

  Other package used in python : math, time, random

  <subsection|Dataset>

  A dataset is built by myself in order to check if algorithm work well when
  meet some edge conditions. Example is list below

  <\code>
    NAME : test

    VERTICES : 2

    DEPOT : 1<next-line>REQUIRED EDGES : 2<next-line>NON-REQUIRED EDGES : 0
  </code>

  <code|VEHICLES : 1<next-line>CAPACITY : 1<next-line>TOTAL COST OF REQUIRED
  EDGES : 1>

  <\code>
    NODES \ \ \ \ \ \ COST \ \ \ \ \ \ \ \ DEMAND<next-line>1 \ \ 2
    \ \ \ \ \ \ \ 13 \ \ \ \ \ \ \ \ \ \ \ \ 1

    END
  </code>

  Since test data all have solution, no-result-condition is not considered
  here.

  <subsection|Solution analysed>

  For some reson, Merge-Split operator is not used in submitted code cause
  some unsolved bugs.

  Perfomance difference before and after introduce mer-split operater is show
  followed.

  Data used in following is data given, since it's hard to show data in
  report.\ 

  \;

  Before

  \;

  <block*|<tformat|<table|<row|<cell|gdb10>|<cell|gbd1>|<cell|egl-s1-A>|<cell|egl-e1-A>|<cell|val7A>|<cell|val4A>|<cell|val1A>>|<row|<cell|275>|<cell|316>|<cell|5412>|<cell|3781>|<cell|280>|<cell|410>|<cell|173>>>>>

  \;

  After

  \;

  <block*|<tformat|<table|<row|<cell|gdb10>|<cell|gbd1>|<cell|egl-s1-A>|<cell|egl-e1-A>|<cell|val7A>|<cell|val4A>|<cell|val1A>>|<row|<cell|275>|<cell|316>|<cell|5308>|<cell|3726>|<cell|283>|<cell|417>|<cell|173>>>>>

  \;

  <section|Conclusion>

  <subsection|Evaluate algorism>

  <subsubsection|Advantage>

  Algorithm use multi-processing to improve performance. By multi-processing,
  algorithmm is easier to get its upper bound. Also, with the introduce of
  merge-split, algorithm's upper bound is higher than normal path-scanning
  algorithm.\ 

  <subsubsection|Defect>

  Since the only random operator is merge-split, algorithm's upper bound is
  still not enough to get best solution. Also, the ratio of path-scanning and
  merge-split is not perfect, for small data, path-scanning will do lots of
  duplicate compute, with is a waste of time.

  <subsection|Space to improve>

  The ratio of path-scanning and merge-split can be improved to get a better
  belance. Also, a memetic algorithm is better to introduce randomness, to
  get best solution.

  <\bibliography|bib|tm-plain|reference>
    <\bib-list|3>
      <bibitem*|1><label|bib-3>Bruce<nbsp>L.<nbsp>Golden<localize| and
      >Richard<nbsp>T.<nbsp>Wong. <newblock>Capacitated arc routing problems.
      <newblock><with|font-shape|italic|Networks>, 11(3):305\U315,
      1981.<newblock>

      <bibitem*|2><label|bib-2>Ke Tang, Yi Mei<localize|, and >Xin Yao.
      <newblock>Memetic algorithm with extended neighborhood search for
      capacitated arc routing problems. <newblock><with|font-shape|italic|IEEE
      Transactions on Evolutionary Computation>, 13(5):1151\U1166,
      2009.<newblock>

      <bibitem*|3><label|bib-1>Wikipedia. <newblock>Arc Routing \V Wikipedia,
      the free encyclopedia. <newblock>2004. <newblock>[Online; accessed
      22-July-2004].<newblock>
    </bib-list>
  </bibliography>

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|3|2>>
    <associate|auto-11|<tuple|3.1|2>>
    <associate|auto-12|<tuple|3.2|2>>
    <associate|auto-13|<tuple|3.2.1|2>>
    <associate|auto-14|<tuple|3.2.2|2>>
    <associate|auto-15|<tuple|3.3|3>>
    <associate|auto-16|<tuple|3.3.1|3>>
    <associate|auto-17|<tuple|3.3.2|3>>
    <associate|auto-18|<tuple|3.3.3|?>>
    <associate|auto-19|<tuple|3.4|?>>
    <associate|auto-2|<tuple|1.1|1>>
    <associate|auto-20|<tuple|3.4.1|?>>
    <associate|auto-21|<tuple|3.4.1.1|?>>
    <associate|auto-22|<tuple|3.4.1.2|?>>
    <associate|auto-23|<tuple|3.4.1.3|?>>
    <associate|auto-24|<tuple|3.4.2|?>>
    <associate|auto-25|<tuple|3.5|?>>
    <associate|auto-26|<tuple|4|?>>
    <associate|auto-27|<tuple|4.1|?>>
    <associate|auto-28|<tuple|4.1.1|?>>
    <associate|auto-29|<tuple|4.1.2|?>>
    <associate|auto-3|<tuple|1.2|1>>
    <associate|auto-30|<tuple|4.2|?>>
    <associate|auto-31|<tuple|4.3|?>>
    <associate|auto-32|<tuple|5|?>>
    <associate|auto-33|<tuple|5.1|?>>
    <associate|auto-34|<tuple|5.1.1|?>>
    <associate|auto-35|<tuple|5.1.2|?>>
    <associate|auto-36|<tuple|5.2|?>>
    <associate|auto-37|<tuple|5.2|?>>
    <associate|auto-4|<tuple|1.3|1>>
    <associate|auto-5|<tuple|2|1>>
    <associate|auto-6|<tuple|2.1|1>>
    <associate|auto-7|<tuple|2.2|1>>
    <associate|auto-8|<tuple|2.3|1>>
    <associate|auto-9|<tuple|2.4|2>>
    <associate|bib-1|<tuple|3|3>>
    <associate|bib-2|<tuple|2|3>>
    <associate|bib-3|<tuple|1|3>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|bib>
      1

      1

      2

      2

      3

      2
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1.<space|2spc>Introduction>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1.<space|2spc>Arc Routing Problem
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2.<space|2spc>Application
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|1.3.<space|2spc>Purpose
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2.<space|2spc>Preliminary
      Problem Description> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1.<space|2spc>Overview of CARP
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|2.2.<space|2spc>Describe
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|1tab>|2.3.<space|2spc>Formulation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <with|par-left|<quote|1tab>|2.4.<space|2spc>Solution-mark format
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3.<space|2spc>Procedure>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10><vspace|0.5fn>

      <with|par-left|<quote|1tab>|3.1.<space|2spc>Introduction
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <with|par-left|<quote|1tab>|3.2.<space|2spc>Data Reading
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12>>

      <with|par-left|<quote|2tab>|3.2.1.<space|2spc>First Part
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13>>

      <with|par-left|<quote|2tab>|3.2.2.<space|2spc>Secand Part
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-14>>

      <with|par-left|<quote|1tab>|3.3.<space|2spc>Distance Calculate:
      Shortest Path Algorithm\VFloyd <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-15>>

      <with|par-left|<quote|1tab>|3.4.<space|2spc>Routes Decide
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-16>>

      <with|par-left|<quote|2tab>|3.4.1.<space|2spc>Path-Scanning
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-17>>

      <with|par-left|<quote|2tab>|3.4.2.<space|2spc>Merge-Split Operator
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-18>>

      <with|par-left|<quote|1tab>|3.5.<space|2spc>Working \]ow
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-19>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4.<space|2spc>Experiments>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-20><vspace|0.5fn>

      <with|par-left|<quote|1tab>|4.1.<space|2spc>Environment
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-21>>

      <with|par-left|<quote|2tab>|4.1.1.<space|2spc>Hardware Environment
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-22>>

      <with|par-left|<quote|2tab>|4.1.2.<space|2spc>Software Environment
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-23>>

      <with|par-left|<quote|1tab>|4.2.<space|2spc>Dataset
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-24>>

      <with|par-left|<quote|1tab>|4.3.<space|2spc>Solution analysed
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-25>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5.<space|2spc>Conclusion>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-26><vspace|0.5fn>

      <with|par-left|<quote|1tab>|5.1.<space|2spc>Evaluate algorism
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-27>>

      <with|par-left|<quote|2tab>|5.1.1.<space|2spc>Advantage
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-28>>

      <with|par-left|<quote|2tab>|5.1.2.<space|2spc>Defect
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-29>>

      <with|par-left|<quote|1tab>|5.2.<space|2spc>Space to improve
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-30>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Bibliography>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-31><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>