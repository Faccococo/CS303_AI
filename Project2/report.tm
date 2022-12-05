<TeXmacs|2.1.1>

<style|ieeeconf>

<\body>
  <doc-data|<doc-title|CS303 Artificial Inteligence Project Report:
  CARP>|<doc-author|<author-data|<author-name|Zitong
  Huang>|<\author-affiliation>
    12012710@mail.sustech.edu.cn

    <date|>
  </author-affiliation>>>>

  <section|Introduction>

  <subsection|Overview>

  The CARP(capacitated arc routing problem) is a typical form of the arc
  routing problem. It is a combinatorial optimization problem with some
  constrains needed<mouse-over-balloon|[1]||left|Bottom>. CARP is proved as a
  NP-hard Problem in \ 1981 by B. L. Golden and R. T.
  Wong<mouse-over-balloon|[2]||left|Bottom>, which means it cannot be solved
  in polynomial time.

  <subsection|Describe>

  CARP can be describe as follows:\ 

  <\quote-env>
    <text-dots>consider an undirected connected graph \<#1D43A\> =
    (\<#1D449\>, \<#1D438\>), with a vertex set \<#1D449\> and an edge set
    \<#1D438\> and a set of required edges (tasks) \<#1D447\> \<subseteq\>
    \<#1D438\>. A fleet of identical vehicles, each of capacity \<#1D444\>,
    is based at a designated depot vertex <math|v<rsub|0>> \<in\> \<#1D449\>.
    Each edge \<#1D452\> \<in\> \<#1D438\> incurs a cost \<#1D450\>\<#1D450\>
    \<#1D452\>\<#1D452\> whenever a vehicle travels over it or serves it (if
    it is a task). Each required edge (task) \<#1D70F\> \<in\> \<#1D447\> has
    a demand \<#1D451\>(\<#1D70F\>) \<gtr\> 0 associated with
    it.<mouse-over-balloon|[1]||left|Bottom>
  </quote-env>

  Unlike other Arc Routing Problem, CARP introduce a capacitate to constrain
  serve-edge number for each node. When capacity filled up, a velhicle can
  serve edges again only after it reach depot edge at least one time.

  <section|BackGround>

  <subsection|Shortest Path Algorithm\VFloyd>

  When doing CARP, a shortest path algorithm is needed to compute distance
  between two node. Algorithm used here is Floyd algorithm, pseudo-code is
  list below:

  <\render-code>
    for every node k in map:

    \;
  </render-code>

  \ 

  <subsection|Path-Scanning>

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

  <subsection|>

  \;

  \;

  \;

  \;

  \;

  \;

  <section|Reference>

  [1]K. Tang, Y. Mei and X. Yao, "Memetic Algorithm With Extended
  Neighborhood Search for Capacitated Arc Routing Problems," in IEEE
  Transactions on Evolutionary Computation, vol. 13, no. 5, pp. 1151-1166,
  Oct. 2009, doi: 10.1109/TEVC.2009.2023449.

  \;

  [2]B. L. Golden and R. T. Wong, \PCapacitated arc routing problems,\Q
  Networks, vol. 11, no. 3, pp. 305\U315, 1981.

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
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.2|?>>
    <associate|auto-4|<tuple|2|?>>
    <associate|auto-5|<tuple|2.1|?>>
    <associate|auto-6|<tuple|2.2|?>>
    <associate|auto-7|<tuple|2.3|?>>
    <associate|auto-8|<tuple|3|?>>
    <associate|footnote-1|<tuple|1|?>>
    <associate|footnr-1|<tuple|1|?>>
  </collection>
</references>