# 🛒 Shopper Spectrum  
## Customer Segmentation and Product Recommendations in E-Commerce

Shopper Spectrum is an end-to-end Machine Learning project that analyzes e-commerce transaction data to understand customer purchasing behavior.

The project uses **RFM Analysis**, **KMeans Clustering**, and **Item-Based Collaborative Filtering** to:

- Segment customers into meaningful groups
- Identify high-value and at-risk customers
- Recommend similar products based on customer purchase behavior
- Provide an interactive Streamlit web application for predictions and recommendations

---

## 📌 Problem Statement

E-commerce platforms generate a large amount of transaction data every day. Analyzing this data helps businesses understand customer behavior, identify valuable customers, improve marketing campaigns, and recommend relevant products.

This project performs customer segmentation using Recency, Frequency, and Monetary value (RFM) analysis. It also builds a product recommendation system using cosine similarity between products.

---

## 🎯 Project Objectives

- Clean and preprocess e-commerce transaction data
- Perform Exploratory Data Analysis (EDA)
- Create RFM features for customer behavior analysis
- Segment customers using KMeans Clustering
- Identify High-Value, Regular, Occasional, and At-Risk customers
- Build an item-based collaborative filtering recommendation system
- Create a Streamlit application for real-time customer segmentation and product recommendations

---

## 🧠 Machine Learning Techniques Used

### 1. Customer Segmentation

Customer segmentation is performed using:

- RFM Analysis
- KMeans Clustering
- StandardScaler
- Log Transformation
- Elbow Method
- Silhouette Score

### 2. Product Recommendation System

The recommendation system uses:

- Item-Based Collaborative Filtering
- Customer–Product Purchase Matrix
- Cosine Similarity
- Product-to-Product Similarity Matrix

---

## 📊 Dataset Information

The dataset contains online retail transaction data.

| Column Name | Description |
|---|---|
| InvoiceNo | Transaction number |
| StockCode | Unique product/item code |
| Description | Product name |
| Quantity | Number of products purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per product |
| CustomerID | Unique customer identifier |
| Country | Customer country |

### Dataset Cleaning Steps

The following preprocessing steps were performed:

- Removed rows with missing `CustomerID`
- Removed rows with missing product descriptions
- Removed duplicate records
- Removed cancelled invoices starting with `C`
- Removed transactions with zero or negative quantity
- Removed transactions with zero or negative unit price
- Converted `InvoiceDate` into datetime format
- Created a new `TotalAmount` feature
```text
TotalAmount = Quantity × UnitPrice
```

## 📸 Outputs

![Home Page](screenshots/Screenshot%202026-06-24%20220456.png)

![Product Recommendations](screenshots/Screenshot%202026-06-24%20220520.png)

![Customer Segmentation](screenshots/Screenshot%202026-06-24%20220532.png)
