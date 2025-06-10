# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Stress Test Supply Chain Networks at Scale
# MAGIC
# MAGIC Numerous past disruptions have exposed the fragility of global supply chains. In response, companies have adopted strategies such as reshoring production and stockpiling critical materials. While these measures aim to improve continuity, they often result in higher costs and increased financial risk. To better anticipate and manage such challenges, a well-cited [paper](https://dspace.mit.edu/handle/1721.1/101782) proposes stress testing supply chains using digital twins—virtual models built from real operational data. By simulating a range of disruption scenarios, businesses can assess potential impacts, identify vulnerabilities and make informed, proactive decisions.
# MAGIC

# COMMAND ----------

# MAGIC %md <img src="images/cartoon.png" alt="Simplified Supply Chain Network" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC ## How does this method work?
# MAGIC
# MAGIC The methodology outlined in the paper—and implemented in this solution accelerator—models the supply chain as a directed graph of materials, plants, and flows. For each node, we capture inventory levels, production capacity, estimated time-to-recover (TTR) after a disruption, and, for product nodes, profit contribution. To simulate the failure of a specific node or a set of nodes, we “remove” the corresponding node(s) from the graph for the duration of their TTR, then solve a linear optimisation problem that reallocates inventory, reroutes materials, and idles plants to minimise lost profit (or another selected metric such as volume or sales). We run this simulation across a comprehensive set of disruption scenarios at scale. Resulting impacts are ranked from negligible to catastrophic, while a complementary time-to-survive (TTS) metric indicates how long operations can continue if a node fails, highlighting sensitivity to TTR assumptions.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why this method for stress-testing?
# MAGIC
# MAGIC First, this method avoids the need for probability estimation, allowing business stakeholders to focus on visible vulnerabilities rather than guessing the likelihood of rare events. Second, because it is optimisation-based and relies only on operational data, it scales to thousands of suppliers and can be refreshed weekly—or even daily. Third, the risk exposure metric uncovers hidden, low-spend suppliers whose failure could severely disrupt production, enabling smarter prioritisation of inventory buffers or dual sourcing. Fourth, the TTS metric highlights where TTR estimates may be unreliable, helping guide supplier discussions. Collectively, these advantages reduce analysis time from years to days and ensure mitigation efforts target the most critical risks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which tools do we use?
# MAGIC
# MAGIC This method is flexible and can be implemented with many technologies. In this solution accelerator we pair the modelling library [Pyomo](https://pyomo.readthedocs.io/en/stable/index.html) with the solver [HiGHS](https://github.com/ERGO-Code/HiGHS). Both are open-source and licensed for commercial use.
# MAGIC
# MAGIC **Pyomo** lets you express linear-optimisation models in clear, algebraic Python syntax, so business rules translate directly into code. It connects to numerous commercial and open-source solvers, enabling rapid prototyping on a laptop and seamless scaling to clusters.
# MAGIC
# MAGIC **HiGHS** provides state-of-the-art dual/primal simplex and interior-point algorithms. It often matches or outperforms commercial solvers on large, sparse LPs, exploits multicore hardware, and supports fast warm-starts.
# MAGIC
# MAGIC To run thousands of distributed scenarios at scale, we orchestrate Pyomo + HiGHS with [Ray](https://docs.databricks.com/aws/en/machine-learning/ray/), a lightweight framework for elastic, fault-tolerant parallel computation.

# COMMAND ----------

# MAGIC %md
# MAGIC The remaining notebooks provide a detailed implementation of the solution and conduct a comprehensive analysis on Databricks.
# MAGIC
# MAGIC * In `01_operational_data`, we generate a synthetic supply chain network along with associated operational data, and review the network's properties and data requirements.
# MAGIC * In `02_stress_testing (small network)`, we demonstrate how to run stress tests on a small supply chain network and analyze the results.
# MAGIC * In `03_stress_testing (large network)`, we scale up the approach to perform stress testing on a larger supply chain network.
# MAGIC * In the optional notebook `04_appendix`, we delve into the mathematical formulation of the optimization problem, define the variables, and discuss key assumptions.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | highspy | Linear optimization solver (HiGHS) | MIT | https://pypi.org/project/highspy/
