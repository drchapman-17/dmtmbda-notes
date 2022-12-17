# Table of Contents
1. [ Data mining .](#dm)  
2. [ Text mining .](#tm)  
3. [ Big Data Analytics .](#bda)  
  

<a name="dm"></a>
# Data Mining
## Business Intelligence
Process of:
- Transforming **raw data** into **useful information** to support effective and aware business strategies.
- Capturing business data and getting the right information to the right **people**, at the right **time**, through the right **channel**.

### Data Warehouse

Main tool to support BI.

BI Platform → ad-hoc infrastructure (hardware and software) necessary to allow flexible and effective business analysis:

- Ad-hoc hardware
- Network Infrastructure
- Databases
- Data Warehouse
- Front-end (Data Visualization)

## Main Concepts of Data Warehousing


- **Data Warehouse (DWH)** = main tool o support BI.
- Optimized repository that stores information for **decision-making process**.
- Specific type of **Decision support systems**.

Increasing number of information → needs more sophisticated solutions than relational databases.

### Advantages of DW systems

- Ability to manage **sets of historical data**
- ability to perform **multidimensional analyses** (accurately and rapidly).
- Based on a **simple model** that can be easily learnt by users.
- Basis for **indicatior-calculationg systems**.

### Data Warehouse (DWH)

**Collection of data** that supports decision-making processes.

- **Subject oriented**: focuses on enterprise-specific contepts (customers, products, sales, etc.).
- **Integrated and costant**: integrates data from different and heterogeneous sources and provides a unified view.
- shows **data evolution over time** and it's **not volatile**: changes to the data in the database are **tracked and recorded** so that reports can be produced showing **échanges over time**. Data is **never written or deleted** (static, read only, retrained for future reporting).

### DWH fields of application

- **Trade**, e.g. sales and claims analysis
- **Financial services**: risk analysis, fraud detection...
- **Transport Industry**: Vehicle management.
- **Telecommunication services**: customer profile analysis, network performance analysis...
- **Healthcare services**: patient admission and discharge analysis.

### Multidimensional Model

**Online Analytical Processing (OLAP)** allows users to iteractively navigate the data warehouse information exploiting the multidimensional model (different levels of aggregation).

E.g. Product → Sub-Category → Category

### Example of OLAP queries
- Which products maximize the profit?
- What is the total revenue per product category?
- What's the relationship between profits hained by two different products?
- Revenue in the laste three yers?

### OLTP and OLAP

**OLTP**:
- Based on **transactions**
- Each transaction reads and writes a few number of records from tables → simple relationships
- OLTP systems have an essential workload core "fozen" in application programs

**OLAP**:
- Interactive Data Processing system for **dynamic multidimensional analyses**.
- Each query involves **huge amount of records** to process a set of numeric data summing up the perfrmance of enterprise
- Worload changes over time

### DataMart

Subset or aggregation of data stored to a primary data warehouse.

Includes set of information pieces relevanto to a specific area (business, corporate, category).

- Used as building blocks while incrementaly developing DWHs.
- Mark out the information required  by a **specific group of users to solve queries**.
- Can deliver better performance → smaller of DWHs.

## OLAP

**Olap Analyses** allow users to interactively navigate the DWH information.
Data is analysed at different levels of aggregation applying OLAP operators.

**Olap Session**: scout the multidimensional model choosing the next operator based on the outcome of the previous ones. It creates a **navigation path** that corresponds to an analysis process for facts (to different points and different detail levels).

*E.g.* Product → Sub-Category → Category

## OLAP Operators

### Roll-up

Causes an **increase** in data aggregation and removes a detail level from a hierarchy.

*e.g.* Counting products by type (Dairy, Beverages)

### Drill-Down

Complement to Roll-up operator. Reduces data aggregation and adds new detail. (e.g. from category to sub-category).

### Slide and dice

Reduces the number of cube dimensions after setting one of the simensions to a specific value (e.g. category='food and beverages'); the dicing operation reduces the set of data being analysed by a selection criterion.

### Pivot

Implies a **change in layouts** → analysing data from different view-points.

*e.g.* using year as a table attribute instead of a column

### Drill-Across

Allows to create a link between concepts in interrelated cubes to compare them.

### Drill-Through

Switches from multidimensional aggregate data to operational data in sources or in the reconciled layer.

## Extraction, Transformation and Loading (ETL)

**ETL** extracts, integrates and cleans data from operational sources to feed the DataWarehouse layer.

### Extraction

Extraction of information from sources (structured or unstructured data).

It can be:

- **Static**: DWH is populated for the first time, Snapshot of operational Data.
- **Incremental**: used to update DWH regularly. It includes **changes** applied to source data since the latest extraction. It is based on:
	- **Timestamp** associated to operational data
	- **Triggers** associated with change transactions

### Cleansing

Used to improve Data quality: standardisizing, correcting mistakes and inconsistencies.

Examples:
- Duplicate Data
- Missing Data
- Unexpected use of fields (e.g. Note used to store a phone number)
- Impossible or wrong values
- Inconsisted values for a single entity.
- Inconsistent values for won individual entity because of typing mistakes.

### Solutions for Data Inconsistencies

- **Dictionary Based Techniques**: used to check correctness of the attribute values based on *lookup tables*. Used for typing mistakes and format discrepancies.
- **Approximate merging**: used when merging data from different sources and a common key is not available (approximate join→ Join based on common attributes, similarity functions→ Affinity functions to calculate similarity between two words).
- **Ad-hoc algorithms**: custom algorithms on specific business rules.

### Transformation

Data from sources is properly transformed to adjust its format to the reconciled schema.

- **Conversion**: changes data types and format
- **Enrichment**: combination of one or more attributes to create information.
- **Separation/concatenation**

### Denormalization

Table redundancy to reduce the number of joins. 

### Loading

Two Different Ways:

- **Refresh**: the DWH is completely rewritten (i.e. older data is replaced). Used in combination with static extraction.
- **Update**: only those changes applied to source data are added to the DWH. Preexisting data is not deleted or modified. Used in combination with incremental extraction.

## DWH Architectures

### Architecture requirements

- **Separation**: analytical and transactional processing should be kept apart
- **Scalability**: hardware and software architectures should be easy to upgrade as the data volume increases.
- **Extensibility**: the Architecture should be able to host new applications and technologies (progressively).
- **Security**: monitoring access essential (security reasons).
- **Administrability**: DWH management should be easy.

### Single Layer Architecture
- Minimize the amount of stored data, removing redundancies
- Source layer = only physical layer available
- DWH implemented as multidimensional view of operational data, created by a Middleware.

*Schema*: 

Source Layer → DWH Layer (Middleware) → Analysis Layer (Reporting/OLAP tools)

| Pros | Cons |
|--|--|
|The occupation of space is minimized.  | No Separation between analytical and transactional processing  |

### Two Layer Architecture

- Separation between Physically available sources and Data Warehouses.
- **Source Layer**: includes a set of heterogenous data sources (internal/external).
- **Data Staging**: data stored should be extracted, cleansed and integrated. (**ETL procedures**).
- **DWH Layer**: information is stored to one logically centralized single repository that can be directly accessed or used as a source for DataMarts. Meta-data repos store information on sources, staging etc.
- **Analysis Layer**: accessed by end-users to create reports, look at dashboarda etc.

### Three Layer Architecture

- **Reconciled Layer**: materializes operational data obtained after integrating  and cleansing source data. 
- Reconciled data layer creates a common reference data model for a whole enterprise and it separates the problems of **source data extraction and integration** from **DWH population**.
- Reconcile data → More redundancy of operational source data.

*Namely*:

... → Data Staging → Reconciled Data → Loading → ...

## Crisp DM

### Standard Process Mode

- Can Data Mining be a push-button technology? → NO
- It's a Process
- Steps + Complex Choies

DM Requires:
- Good mix of tools and skilled analysts
- sound methodology
- project management
- **process model to manage interactions**

Standardisation:
- Common reference point for discussions
- Common understanding between designers and customers

### Business Understanding

- Reformulate the problem in many ways, as necessary
- Scenario definition
- Iterative refinement 

*Tasks*:

- **Determine Goals**:
	- Business Objectives
	- Background BO
	- Business success Criteria
- **Assess Situation**:
	- Inventory of *Resources*
	- Requirements Assumptions and constraints
	- Risk and contingencies
	- Costs and benefits
- **Goals**:
	- Data Mining Goals
	- DM Sucecss Criteria
- **Produce Plan**:
	- Project Plan
	- Initial Assessment of Tools and Techniques

### Data Understanding

- **Which raw data are available**?
	- They match problem needs
	- Usually collected for different purposes
- **At which cost**?
	- External data might not be free
	- Ad-hoc campaigns are costly
- **Possible forks** in the project choices

*Tasks:*

- Initial Data Collection → Report
- Data Description → Report
- Data Exploration → Report
- Data quality assessment → Report

### Data Preparation

- Some analysis techniques may require data transformations
- Some transformations improve result quality
- Data leaks

*Tasks*:
- Data set
- Select Data → Rationale for Inclusion/Exclusion
- Clean Data
- Construct Data (derived attributes + record generation)
- Integrate data (merging)
- Format Data

### Modeling

Capturing patterns hidden in data.

*Tasks*:

- Select **Modeling Technique** (+ assumptions about it)
- Generate **Test Design**
- **Build** Model (parameters, model, description)
- **Assess** Model (Revised Parameter settings).

### Evaluation

- Rigorous assessment of the results of the data mining process
- *qualitative* and *quantitative* bias
- evaluate model **confidence**
- estimate the **business impact**

*Tasks*:
- **Evaluate Results** (w.r.t. Business Success Criteria)
- Review Process
- Determine Next Steps

### Deployement

Results of DM Process (models) are used in software systems.

*Tasks*:
- Plan Deployement
- Plan **Monitoring and Maintenance**
- Produce Final Report
- Review Project




## Conceptual Modeling: Dimensional Fact Model (DFM)

DFM: conceptual mkodel created specifically to function as data martk design support.

Graphic and based on the multidimensional model.

### Basic Concepts

|Concept|Description|Example|
|----------|--------------|------|
|**Fact**| Concept relevant to decision Making Processes. Models a set of events taking place within a company| Sales, Purchases, orders|
|**Measure**|Numerical property of a fact (Quantitative).|Quantity, revenue, discount|
|**Dimension**| Fact property with a finite domain. Describes analysis coordination| Date, product, store|
|**DImensional Attribute**| Dimensions and other possible attributes. Discrete values.| Category (of product), month|
|**Hierarchy**|Directed tree whose nodes are dimensional attributes and whose arcs model *many-to-one* associations between dimensional attribute pairs|Date→Month→Year|
|**Primary event**|Particular occurrence of a fact, identifiad by n-ple made up of a value for each dimension.||
|**Secondary Event**| Given a set of dimensional attributes each n-ple of their values identifies a seondary event which aggregates all of the corresponding primary events.||

### Additivity

Aggregation requires the definition of a suitable operator to compose the measure values that mark primary events into values to be assigned to **secondary events**.

A measure is called **additive** along a dimension when you use the **SUM** operator to aggregate its values along the dimension hierarchy.


Measure classification:

- **Flow Measures**: refer to a time-frame → evaluated cumulatively (e.g. quantity Sold).
- **Level Measures**: evaluated at particular times (e.g. stock in inventory).
- **Unit Measures**: level measures but expressed in relative terms (e.g. unit price).

### Aggregation

- **Distirbutive**: aggregates from partial aggregates (sum, min, max).
- **Algebraic**: uses additional information from a finite number of support measures (e.g. AVG).
- **Holistic**: calculating aggregates from partial aggregates via an invinite number of support measures (e.g. RANK).

### Descriptive Attributes

Used to give **additional information** to a specific dimensional attribute (not used as aggregation criteria). e.g. Address.

### Cross Dimensional Attributes

Dimensional or descriptive attributes whose values are defined by the combination of **two or more dimensional attributes** (e.g. VAT on product category in a certain country).

### Convergence

**Two or more arcs** belonging to the same hierarchy and ending at the same dimensional attribute.

### Shared hierarchy

All **descendant of a shared attribute** are shared too.

### Multiple/Optional Arcs

Like DBMS schema.

### Incomplete hierarchy

One or more levels of aggregation prove missing in some instances (because they are not known or defined).

### Recursive Hierarchy

Parent child Relationship among the levels.

### Logical Design

- Star Schema
- Snowlake schema

## Data Lakes

### Dark Data

- Aquired and stored through computer-based operations
- Not used in decision-making process

### What's a Data Lake

- **Repository** o data stored in **natural/raw** format (object blobs or files)
- **Dive anywhere, flexible access, schema on read**
- **Quickly ingests anything** (no schema enforced on write).
- **All data in one place**
- **Low cost scalable**
- **Future Proof**

### Requirements
- - Wide range of analytics use cases
- Data in hands of the business users
- Flexible and scalable data srchitecture
- 20% structured (DW/BI) over 80% unstructured
- Short time to value
- Holistic metadata management, governance, security monitoring

### Features and use cases

- **Data Warehouse** → Mission critical, low latency, insight apps
	- More expensive, use cases specific data, less latency, **more governance**, **higher data quality**
- **Data Lake**→ Staging area, data Mining, searching, proofing, cataloging
	- **less expensive**, More latency, Less governance, Lower Data quality.

### Data Lake User needs are different

- **Business Users** → use pre-configured dashboards and reports.7
- **Business Analysts** → use self-service BI and ad-hoc analytics (build own models to provide insights).
- **Data Scientists, Engineers, App Developers** → Perform statistical Analysis, ML training, BDA to identify trends, solve business problems, optimize performance.



<a name="tm"></a>
# Text Mining


## Text Processing
- **Structured Data**:  Relational Model
- **Unstructured Data**: Data without models or schemas that can semantically describe them.
	- 99% of web pages, emails, forums, blogs, pubmed, legal documents...
- **Semi structured Data**: XML, RDF, OWL. Partially described by some model
### Information Retrieval

- Methods and algorithms used to search for relevant docs w.r.t. user queries in repositories of *unstructured docs*.

Example:

```sql
SQL: Select * from CUSTOMERS where NAME like "%Business%Intelligence%"
```

- IR should reply to more advanced queries s.a.:
	- Retrieving focs retrieving terms 'Information? adjacent to 'Retrieval' or 'Relevance' withour 'Protocol'.

- **Data selection** → First step after defining Data Mining Goals
	- IR offers efficient methods for **representation and selectionéé of unstructured data.

## Document Representation

###  Binary Vectors
- 0/1 for each term and document
- → AND bitwise among vectors (more efficient than numerical data).
### **Boolean retrieval model** 

→ query expressive power based on boolean proposition. 

### Limits of Vector representation

- Doesn't work well with high dimensions (high number of documents and terms).
 - We are mostly working with **sparse matrices**.

### Solution: Inverted Index

- $\forall t: term$, stores a list of all docs containing $t$.
	- Each doc is defined by a *docid*.

- Extended to → Inverted index with dynamic structures:
	- list and docID stored on disk
	- Dictionary stored in ram (smoller than postings

### Boolean query optimization with inverted index

- Visiting the lists in increasing order of frequency (number of docs containing them) → starting from the **shorter list for better efficiency**.



## Dimensionality Reduction
- Reducing the number of features allows the application of a greater number of ML tools.
- Assumption: Datasets contain redundant or irrelevant variables for training.
- Two areas of research:
	- Feature Selection: selection of a subset of features (data representation unchanged)
	- Feature Extraction: transformation of the data representation computing a reduced subset of features.

Easiest feature selection in TM: TF-IDF (e.g. n terms with best tf-idf).

### 2 methods
- Goal: estimate feature utility in the classification task:
	1. Mutual Information: Measures the reciprocal Dependency between two variables (how much information they share)
		 - For each class the terms with Higher MI are selected
	2.  Chi square test:
		- popular statistical hypothesis test in many domains
		- Indicates whether the difference between observed and expected data is a result of chance.

## Mutual Information

Calculates "how much information a term *t* and a *class* share (number of bits".

If *t* is independent from the clas, then MI is 0.

- *U*: random variable related to the document (if $e_t=1$ then *t* is in the document, else 0)
- *C* is the random variable related to the class and if  $e_c=1$ the document is in the class c, else $e_c=0$.

$I(U,C)=\sum\limits_{e_t \in \{1,0\}}\sum\limits_{e_c \in \{1,0\}}P(U=e_t, C=e_c)log_2\frac{P(U=e_t,C=e_c)}{P(U=e_t)P(C=e_c)}$

Based on MLE.

## Chi-Square Test

- Test to verify independenve between two events A and B
- A and B are independent 
	- if $P(A,B) = P(A)P(B)$ 
	- or if $P(A|B) = P(A) \and P(B|A) = P(B)$

## Latent Semantic Analysis

- Transform the terms-docs matrix by bringing out latent semnatic associations between terms and documents.
	- Maps the matrix in a new vector space with **lower dimension** which ignores irrelevant details
	- Latent semantic association are superior order association based on lower lexical match among terms.
-  Transformed space: terms semantically similar or associated are placed in neighbouring positions.
- *Latent Semantic Indexing*: space used in information retrieval as the index for similarity searches.

**SVD factorization**→ dimentionality reduction through svd (low rank approximation).

U→ Similarity Between Words
V→ Similarity Between Documents

**LSA uses**:
- Collaborative Filteringand Recommendation Systems
- Opinion Mining & Sentiment Analysis (factorized matrix thatcontains opinions and users, LSAcan better lead to user grouping on the base of their opinion)
- Speech-to-text and natural language understanding
- More powerful data clustering

**LSA limits**:
- Cannot directly catch the polysemy, i.e. multi-meaning words:
	- Term vector of different semantic meanings = average → doesn't fully capture polysemy
	- Words with more meaning without a prevalence are positioned between topic clouds.

## Transformers, Efficient Attention and Vision-Language Retrieval Model

### Attention mechanism

- Data $\{x_1,...,x_m\}$ and labels $\{y_1,...,y_m\}$
- Estimate value label $y$ for new data $x$
- Better Idea: predicting label $y$ of new data $x$ by weighting labels according to locations
- $f(x) = \frac{1}{m}\sum\limits_{1}^{m}\phi(x,x_i)\cdot y_i$
- where $\phi(x,x_i)$ is a kernel function (e.g. gaussian)
- Deep learning version: learn weighting function
- Attention allows the decoder to focus (at each timestep) on the most important encoder input
- implemented providing the decoder with the **weighted sum of all the encoder outputs at each timestep**
- Attention weights are learned from a neural network called **attention layer**.

### Concatenative and multiplicative attention

- **alpha weights** = result of a softmax function in the last layer: $a_{(t,i)}=\frac{exp(e_{(t,i)})}{\sum_j exp_{(t,i)}}$
- **Concatenative attention**:
	- $e_{(t,i)} = v^\intercal tanh(W[h_{(t)};y(i)])$
	- $v$= scaling parameter and $;$ = concatenatate operator
- **Multiplicative attention**:
	- $h_{(t)}^\intercal Wy(i)$
	- faster and more space-efficient
	- $W$ matrix could also be non-trainable and set to the identity matrix. (dot product attention)

### Transformer: Input + positional embeddings

Input:
- Matrix $Z$ obtained by transforming inputs into input embeddings
- Given $L$ = input length, $d$, embedding size, $Z=L\times d$

**Positional embeddings**:
- $i$ = position of the word in the sentence
- $j$ = index in the embedding
- $M$ = arbitrary coeff.
- Matrix $P$
- $P_{i,j}$ =
	- $sin(\frac{i}{M\frac{j}{d}})$ if j is even
	- $cos(\frac{i}{M\frac{j-1}{d}})$ if j is odd

## Optimizations for the Attention mechanism

- **Attention Problem** → Quadratic space-time complexity.
- *Most popular strategies to solve the problem*:
	- **Fixed/Learnable Patterns** → Attention only between blocks of size $B < < L$ of the input
	- **Memory** → use additional "memory" in the form of global tokens that allow to aggregate information and reduce the number of attended tokens.
	- **Low-Rank Methods** → use a low-rank approximation of the attention matrix.  It projects the vectors into a lower dimensional space.
	- **Kernels** → use kernel functions to express attention without calculting the $L \times L$ matrix.
	- **Recurrence** → Fixed patterns mehanism but introduce recurrent connections between blocks.

Examples:

- **Performer** → Kernel based solution, rewrites attention formulation. Avoids computing an $A$ matrix using FAVOR+ which approximates kernel K (Keys).
- **Big Bird** → model with linear complexity which uses a memory mechanisms. Approximates attention using: **random attention** (random links between nodes), **window attention** (neighbours connection), **global attention** ( $g$ nodes are completely connected).
- **Reformer** → $O(LlogL)$, approximates attention using *learnable patterns* through **Locality Sensitive Hashing**. Uses shared parameters between $W_Q$ and $W_K$.
- **Transformer-XL** → processes longer documents storing outputs in a cache to be resued as extended context for *keys* and *values*.

## Deep Metric Learning

- Metric learning → find a latent embedding space that allows to separate data according to a given metric
- 2 types:
	- **Supervised** → Labeled data, objective: learn a metric that brings the records belonging to the same class closer.
	- **Weakly Supervised** → No lables, but classes can be inferred from the DS.

### Objective

Find a mahlanobis matrix M s.t. taking two similar elements, (a,p) and one dissimilar element (n) we have that 

$d_M(a,b) < d_M(a,n)$ 

where $d_m(x,y) = \lvert Wx - Wy \rvert_2, M= W\intercal W$

In **Deep Metric Learning** the objective is to learn the weights for the transformation $W_\theta$ .

### Loss Functions

- **Constractive Loss** → Either the squared Mahalanobis distance if the elements are of the same class, or max(0, alpha - Squared distane) if they are of different classes. Makes so that inputs with different labels to have a distance grater than a margin alpha.
- **Triplet Loss** →  given two elements of a class and one of a different class the loss is defined as the maximum between 0 and the squared distance between the anchor and the positive class, minus the squared distance between the anchor and the negative example plus alpha. This makes so that the distance between anchors and positives to be smaller than the distance between anchors and negatives.


<a name="bda"></a>
# Big Data Analytics

## Hadoop Ecosystem Overview

- **HDFS** (Hadoop Distributed File System): distributed filesystem capable of operating reliably on a large number of servers realised with commodity, unreliable hardware.
- **HBase**: key-value (non-relational) store which uses HDFS as a filesystem.
- **YARN** (Yet another resource Negotiatior): system for resource management in a computer cluster. Allows execution of distributed programs in an Hadoop Cluster.
- Recent Developements: SQL-like queries, stream processing, Machine learning programs which require a main memory workspace (MapReduce doesn't offer it).

## MapReduce

- Map and reduction are **functional programming** operations.
- MR is a programing model for batch parallel procesing of large datasets: Map and Reduce operations are executed in parallel on the distributed file system.
- Both functions are now available in many programming languages.

### RDBMS vs MapReduce

- **RDBMS**: capable of performing distributed computation (e.g. distributed query answering).
- They are optimised for **short accesses to small amounts of data**.
- Sequential access has improved faster than random access, so MapReduce is faster than RDBMS for **whole dataset reads and writes**.

### HPC vs MapReduce: HPC

**HPC**→ Focuses on distributing computation across a cluster o computers for CPU-bound jobs (simulation, linear algebra).
- Programmers explicitly control the subdivision of data and work into non-overlapping chunks (processed by distinct computers)
- Coordination: Explicitly controlled by message-passing primitives
- Failures during computation: **Handled by the program**

In contrast MapReduce focuses on **data-intensive jobs** (e.g. complex transformations and aggregations).
- No explicit data and work partitioning → Data is local to the computer which will process them
- No message passing: handled automatically by MapReduce
- Node Failures: **Automatically handled**

### MapReduce

- $map(f,A)$ applies a function $f$ to a collection of objects $A$ = $\{f(x) : x \in A\}$.
- $reduce(f,A)$ applies a function f of **two arguments** to a sequence of objects, accumulating intermediate results until the sequence is reduced to a single value: $f(...f(f(a_1,a_2),a_3)...a_n)$ with $A=(a_1,...,a_n)$.
- In **MapReduce**: computation = sequence of alternating map and reduce function calls which operate in parallel on distributed data.

### MapReduce Programming Model

- Set of key/value pairs as an input → key/value pairs as an output
- Programmer: specifies map and reduce functions
- Map → Intermediate key/value pairs
- Inermediate → Grouped or shuffled by the values of the intermediate key → Passed to reduce function
- Reduce merges the values into a smaller set of values.

### Google's Implementation: Map

- Dataset is partitioned into a number of **splits**  and **copies of the program** are executed on the nodes of the cluster.
- **Master**= distinguished copu. It **assigns** map and reduce tasks to **worker copies**.
- Each map worker reads a split, identifies key/values pairs and passes them through the map function → intermediate key/value pairs are stored in main memory.
- Intermediate pairs are written to the local disk from main memory at intervals, partitioned into **labeled regions** (unique label throughout the cluster).
- The region locations are sent to the master → it communicates them to the **reduce workers**.

### Google's Implementation: Reduce

- Each **reduce worker** collects intermediate key/value pairs from the regions (might be in multiple mappers' disks) → pairs are sorted by key.
- Each reduce worker scans the key-sorted pairs and collects values corresponding to the same key → then it sends them to the user **reduce function** and procedes with the next key.
- The function processes the key and values and writes the result to disk.
- After the last task has terminated the master passes control to the user program.

## HDFS

Designed to achieve:
- **Resilience**: automatic recovery from hardware failures.
- **Streaming Access**: High throughput rather than low latency (Batch applications).
- **Large Datasets**: GB~TB and High Bandwidth.
- **Write-Once, Ready Many**: Files are written and once closed they can be truncated/appended
- **Local Computation**: Support for moving app close to the data.
- **Portability**: different platforms.

## HDFS Architecture

- Master/Slave Architecture
- **NameNode** = Master
	- Manages namespace
	- Regulates client file access
	- Responsibilities:
		-  Associates blocks to DataNodes
		- File namespace operations (open, close, rename)
- **DataNodes** = Slaves
	- 1-1 to Cluster Nodes
	- Manage storage of the nodes they run on
	- Responsiilities:
		- Block management: create, delete replicate
		- Read and write requests from clients
- Data → contained in files → Split into blocks → Multiple Data Nodes

## Resource Management in Hadoop (YARN)

- Starting with hadoop 2
- Nor normally under user control, but YARN provides an API which exposes cluster resources
- Operates through two deamons:
	- **resource manager**: 
		- one per cluster
		- creates and manages **containers**: processes with a bounded amount of each computational resource
	- **node manager**:
		- run on evey node
- The process (applciation client) sends a request to the resource manager to start an **application master**
- To fulfil the requeste: the resource manager contacts a **node manager** and asks it to run a container and the application master as a process inside it.

### Application master

- AM algorithm is Application-dependent
-  Simplest form: performs a computation using the container's allocated resources
- A **MR** application sends requests to the resource manager to obtain other containers to run a *distributed computation*
- Requests for containers can specify:
	- Amount of resources (for the container)
	- Locality constraints (e.g. specific nodes). This can't be always be satisfied → Yarn tries to fin the closest node.
- YARN: Flexible → allows for computation to be performed as close to the data as possible.

## Spark

### Limitations of MapReduce
**MR** assumes that processng can be expressed as an *acyclic flow of data throigh a series of operations*.

However, many data processing algorithms violate such an ssumption by keeping working-sets between two or more iterations (g.g. machine learning algorithms require values to be preserved between iterations).
- MapReduce → results must be stored on disk at the end of a job.
- Each iteration would be implemented as a MapReduce job → each job stores its result on disk for the next job.
- Reading and writing to disk are slower than reading and writing to RAM

### The Spark Project
Cluster computing platform for speed* and *polyvalency*.

- In-Memory Computation
- More efficient than MapReduce (also for disk computation).
- Workloads: Batch applications, iterative algorithms, interactive queries, streaming applications
- API for Java, Python, R, Scala

### Architecture
- **Spark Core**: Task, memory, fault-recovery, storage
- **Spark SQL**: SQL and HQL interface
- **Spark Streaming**: live streams of data
- **MLlib**: Machine Learning Library
- **GraphX**: Graph manipulation, parallel computations
- **Cluster Managers**: Standalone, YARN, Mesos

### Processing Modes
**Spark Application**= *driver program* which coordinates a set of *executor processes*
- When *driver* and éexecutors* run on different nodes of a cluster, Spark is running in **cluster mode**:
	- Cluster manager which informs about the availability of resources
	- Intended for *production*
- When the driver and the executors run on the same machine: **Local Mode**
	- Used fpr exploring data, experimenting, debugging

### Driver, worker, executor
- **Driver**:
	- Contains the *Main Function*
	- Defines *Distributed datasets*
	- Performs *operations* on datasets
- **Worker**:
	- Process that remains in exeution and keeps the datasets distributed in RAM
	- Normally **only one for each worker node**
- **Executor**
	- Performs *tasks*
	- Dialogue with a *single* application
	- Running for the duration of the application
	- Possibly multiple executors for each worker

### DataFrames, Datasets

- Both are data collections which can be represented as a table (similar to SQL)
	- Schema with column names and column types
	- Values of a column have the same type of the column
	- all columns have the same length
- DFs are available for all languages. 
	- Confermance = runtime
- Datasets are available for Scala and Java
	- Confermance = compile time

### Dataframe creation
```python
source = "Spark/Data/measurements.json"
```
The JSON contains row like data:
```json
{ "Code": 88118, "Date": "2015-10-30", "Pollutant": "Ozone", ɭɭ.}, 
{ "Code": 927170, "Date": "2015-10-08", "Pollutant": "Particulate matter", ɭɭ.},
```
A dataframe can be created by:
```python
df = spark.read.format("json").load(source)
df.show()

```

| Code |Concentration  | Date|Pollutant|Station|
|--|--|--|--|--|
| 88118| 79|2015-10-30| Ozone| Rimini|
|927170| 25|2015-10-08|Particulate matter|Ravenna|
|190115| 59|2015-02-04| Ozone| Rimini|

### DataFrame Schema-on-Read

Infers the following schema:
```python
df.schema 
```
```shell
StructType(List(StructField(Code,LongType,true), 
StructField(Concentration,LongType,true), 
StructField(Date,StringType,true),
StructField(Pollutant,StringType,true), 
StructField(Station,StringType,true))) 
```
```shell
df.printSchema() 
```

```cmd
root 
	|-- Code: long (nullable = true) 
	|-- Concentration: long (nullable = true) 
	|-- Date: string (nullable = true) 
	|-- Pollutant: string (nullable = true) 
	|-- Station: string (nullable = true)

```

### Columns
- Column manipulation requires importing *col* or *column* libraries

```python
from pyspark.sql.functions import col 
from pyspark.sql.functions import column
```
- A referene to a column is
```python
col("Pollutant") # or 
column("Pollutant")
```
- Such references do not correspond to values outside a DF.

### Expressions

- Set of transformations on values of a ROW
- Based on constants, operators, columns
- Results = exactly the same of a semantically equivalent DataFrame expression

```python
from pyspark.sql.functions import expr 
expr("Concentration < 50 and Concentration > 10") # SQL-like 
(col("Concentration")) < 50 & (col("Concentration") > 10) # DataFrame 
expr("Concentration * .9 + 0.01") # SQL-like 
col("Concentration") *.9 + 0.01 # DataFrame

```

### Rows
- Manipulation requires importing "Row" from pyspark.sql
- Row created with the Row function


```python
from pyspark.sql import Row 
meas = Row(88118,"2015-10-30","Ozone","Rimini",79)
```
Accessible as Python slices:
```python
meas[0] 
meas[1:4] 
```
```shell
88118 
('2015-10-30', 'Ozone', 'Rimini')
```
