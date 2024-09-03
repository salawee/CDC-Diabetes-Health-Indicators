---
title: "Predictive Analytics in Public Health: Early Detection of Diabetes Using BRFSS Data"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(readr)
library(caret)
library(ggplot2)

Introduction
Diabetes is a significant public health concern in the United States, affecting millions and imposing substantial economic burdens. This project aims to utilize the Behavioral Risk Factor Surveillance System (BRFSS) data to develop predictive models for early detection of diabetes and pre-diabetes states.

Data Loading
