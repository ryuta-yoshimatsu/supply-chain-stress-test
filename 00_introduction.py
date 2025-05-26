# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Stress Testing Supply Chain Networks at Scale
# MAGIC
# MAGIC In the recent trade war, governments have weaponized commerce through cycles of retaliatory tariffs, quotas, and export bans. The shockwaves have rippled across supply chain networks and forced companies to reroute sourcing, reshore production, and stockpile critical inputs—measures that extend lead times and erode once-lean, just-in-time operations. Each detour carries a cost: rising input prices, increased logistics expenses, and excess inventory tying up working capital. As a result, profit margins shrink, cash-flow volatility increases, and balance-sheet risks intensify.
# MAGIC
# MAGIC Was the trade war a singular event that caught global supply chains off guard? Perhaps in its specifics—but the magnitude of disruption was hardly unprecedented. Over the span of just a few years, the COVID-19 pandemic, the 2021 Suez Canal blockage, and the ongoing Russo-Ukrainian war each delivered major shocks, occurring roughly a year apart. These events, difficult to foresee, have caused substantial disruption to global supply chains. 
# MAGIC
# MAGIC What can be done to prepare for such disruptive events? Instead of reacting in panic to last-minute changes, can companies make informed decisions and take proactive steps before crises unfold? A well-cited research paper by MIT professor David Simchi-Levi presents a compelling, data-driven approach to this challenge. At the core of his method is the creation of a digital twin of the supply chain—built using real operational data—to simulate a wide range of disruption scenarios. By analyzing how the network responds, companies can assess potential impacts, uncover hidden vulnerabilities, and identify redundant investments. This process, known as stress testing, has become widely adopted across industries.
# MAGIC
# MAGIC The remainder notebooks provide a detailed implementation of the solution and perform a comprehensive analysis on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="images/cartoon.png" alt="Simplified Supply Chain Network" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | pyomo | An object-oriented algebraic modeling language in Python for structured optimization problems | BSD | https://pypi.org/project/pyomo/
# MAGIC | coinor-cbc | COIN-OR Branch-and-Cut solver | Eclipse Public License - v 2.0 | https://github.com/coin-or/Cbc
