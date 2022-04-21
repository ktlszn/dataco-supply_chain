# **Improving Supply Chain Management by Machine Learning Applications: DataCo Global Case Study**

# **Abstract**
In today’s globalized world, supply chains are the foundations of international trade. By variety of newly developed data collection and machine learning methods, application of big data analytics on supply chains become more accessible and widely utilized. Prediction of the status of goods and shortening supply chains are still stand out as a few of main problems in supply chain management. In this project, I investigate possible practices to explore supply chain data by analyzing DataCo Global Supply Chain dataset and apply multiple machine learning models to improve supply chain management.
## **Keywords**
Machine learning; predictive analytics; classification; regression; supply chain management 
## **Research Questions**
1) How to detect late deliveries in supply chains before the order delivered? 
1) Is it possible to predict the profitability of orders before orders are realized?
1) How can shipping modes affect supply chains and delivery of goods?

**Tools:** Python, Jupyter Notebook, Sklearn, Pandas, Numpy, Matplotlib

**GitHub repository:** <https://github.com/ktlszn/dataco-supply_chain>
## **Data Dictionary**
### **Summary of Categorical Attributes**
We have total 24 categorical attributes in the dataset. Categories are summarized below in terms of description, number of levels and examples of levels.
### *Table 1. Categorical Attributes*

|**Attribute**|**Description**|**No of Levels**|**Categories**|
| :- | :- | :- | :- |
|**Type**|Type of transaction made|4|<p>DEBIT: 69295</p><p>TRANSFER: 49883</p><p>PAYMENT: 41725</p><p>CASH: 19616</p>|
|**Delivery Status**|Delivery status of orders|4|<p>Late delivery: 98977</p><p>Advance shipping: 41592</p><p>Shipping on time: 32196</p><p>Shipping canceled: 7754</p>|
|**Category Name**|Description of the product category|50|<p>Cleats: 24551</p><p>Men's Footwear: 22246</p><p>Women's Apparel: 21035</p><p>Indoor/Outdoor Games: 19298</p><p>Fishing: 17325</p>|
|**Customer City**|City where the customer made the purchase|563|<p>Caguas: 66770</p><p>Chicago: 3885</p><p>Los Angeles: 3417</p><p>Brooklyn: 3412</p><p>New York: 1816</p>|
|**Customer Country**|Country where the customer made the purchase|2|<p>EE. UU.: 111146</p><p>Puerto Rico: 69373</p>|
|**Customer Email**|Customer's email address|1|XXXXXXXXX|
|**Customer Fname**|Customer first name|782|<p>Mary: 65150</p><p>James: 1835</p><p>Robert: 1759</p><p>Michael: 1680</p><p>David: 1625</p>|
|**Customer Lname**|Customer last name|1109|<p>Smith: 64104</p><p>Johnson: 989</p><p>Brown: 909</p><p>Williams: 869</p><p>Jones: 859</p>|
|**Customer Password**|Masked customer key|1|XXXXXXXXX|
|**Customer Segment**|Types of Customers|3|<p>Consumer: 93504</p><p>Corporate: 54789</p><p>Home Office: 32226</p>|
|**Customer State**|State to which the store where the purchase is registered belongs|46|<p>PR: 69373</p><p>CA: 29223</p><p>NY: 11327</p><p>TX: 9103</p><p>IL: 7631</p>|
|**Customer Street**|Street to which the store where the purchase is registered belongs|7458|<p>9126 Wishing Expressway: 122</p><p>4388 Burning Goose Ridge: 117</p><p>4720 Noble Hills Wynd: 116</p><p>2878 Hazy Wagon Thicket: 113</p><p>398 Emerald Grove: 109</p>|
|**Department Name**|Department name of store|11|<p>Fan Shop: 66861</p><p>Apparel: 48998</p><p>Golf: 33220</p><p>Footwear: 14525</p><p>Outdoors: 9686</p>|
|**Market**|Market to where the order is delivered |5|<p>LATAM: 51594</p><p>Europe: 50252</p><p>Pacific Asia: 41260</p><p>USCA: 25799</p><p>Africa: 11614</p>|
|**Order City**|Destination city of the order|3597|<p>Santo Domingo: 2211</p><p>New York City: 2202</p><p>Los Angeles: 1845</p><p>Tegucigalpa: 1783</p><p>Managua: 1682</p>|
|**Order Country**|Destination country of the order|164|<p>Estados Unidos: 24840</p><p>Francia: 13222</p><p>M�xico: 13172</p><p>Alemania: 9564</p><p>Australia: 8497</p>|
|**order date (DateOrders)**|Date on which the order is made|65752|<p>4/29/2017 10:11     5</p><p>7/16/2015 14:47     5</p><p>9/13/2017 12:17     5</p><p>11/21/2016 23:12    5</p><p>6/5/2016 17:18      5</p>|
|**Order Region**|Region of the world where the order is delivered |23|<p>Central America: 28341</p><p>Western Europe: 27109</p><p>South America: 14935</p><p>Oceania: 10148</p><p>Northern Europe: 9792</p>|
|**Order State**|State of the region where the order is delivered|1089|<p>Inglaterra: 6722</p><p>California: 4966</p><p>Isla de Francia: 4580</p><p>Renania del Norte-Westfalia: 3303</p><p>San Salvador: 3055</p>|
|**Order Status**|Order Status |9|<p>COMPLETE: 59491</p><p>PENDING\_PAYMENT: 39832</p><p>PROCESSING: 21902</p><p>PENDING: 20227</p><p>CLOSED: 19616</p>|
|**Product Image**|Link of visit and purchase of the product|118|http://images.acmesports.sports/Perfect+Fitness+Perfect+Rip+Deck                                 24515|
|**Product Name**|Product Name|118|<p>Perfect Fitness Perfect Rip Deck: 24515</p><p>Nike Men's CJ Elite 2 TD Football Cleat: 22246</p><p>Nike Men's Dri-FIT Victory Golf Polo: 21035</p><p>O'Brien Men's Neoprene Life Vest: 19298</p><p>Field & Stream Sportsman 16 Gun Fire Safe: 17325</p>|
|**Shipping date (DateOrders)**|Exact date and time of shipment|63701|<p>10/25/2016 23:46    10</p><p>3/13/2017 3:26      10</p><p>2/1/2015 2:35       10</p><p>8/7/2016 20:33      10</p><p>4/24/2016 2:18      10</p>|
|**Shipping Mode**|The following shipping modes are presented |4|<p>Standard Class: 107752</p><p>Second Class: 35216</p><p>First Class: 27814</p><p>Same Day: 9737</p>|

### **Summary of Numerical Attributes**
We have total 29 numerical attributes in the dataset. All attributes are summarized below in five-number summary as well as value count, mean, and standard deviation.
### *Table 2. Numerical Attributes*

|**Attribute**|**Description**|**count**|**mean**|**std**|**min**|**25%**|**50%**|**75%**|**max**|
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
|**Days for shipping (real)**|Actual shipping days of the purchased product|180519|3.5|1.6|0.0|2.0|3.0|5.0|6.0|
|**Days for shipment (scheduled)**|Days of scheduled delivery of the purchased product|180519|2.9|1.4|0.0|2.0|4.0|4.0|4.0|
|**Benefit per order**|Earnings per order placed|180519|22.0|104.4|-4275.0|7.0|31.5|64.8|911.8|
|**Sales per customer**|Total sales per customer made per customer|180519|183.1|120.0|7.5|104.4|164.0|247.4|1940.0|
|**Late\_delivery\_risk**|Categorical variable that indicates if sending is late (1), it is not late (0).|180519|0.5|0.5|0.0|0.0|1.0|1.0|1.0|
|**Category Id**|Product category code|180519|31.9|15.6|2.0|18.0|29.0|45.0|76.0|
|**Customer Id**|Customer ID|180519|6691.4|4162.9|1.0|3258.5|6457.0|9779.0|20757.0|
|**Customer Zipcode**|Customer Zipcode|180516|35921.1|37542.5|603.0|725.0|19380.0|78207.0|99205.0|
|**Department Id**|Department code of store|180519|5.4|1.6|2.0|4.0|5.0|7.0|12.0|
|**Latitude**|Latitude corresponding to location of store|180519|29.7|9.8|-33.9|18.3|33.1|39.3|48.8|
|**Longitude**|Longitude corresponding to location of store|180519|-84.9|21.4|-158.0|-98.4|-76.8|-66.4|115.3|
|**Order Customer Id**|Customer order code|180519|6691.4|4162.9|1.0|3258.5|6457.0|9779.0|20757.0|
|**Order Id**|Order code|180519|36221.9|21045.4|1.0|18057.0|36140.0|54144.0|77204.0|
|**Order Item Cardprod Id**|Product code generated through the RFID reader|180519|692.5|336.4|19.0|403.0|627.0|1004.0|1363.0|
|**Order Item Discount**|Order item discount value|180519|20.7|21.8|0.0|5.4|14.0|30.0|500.0|
|**Order Item Discount Rate**|Order item discount percentage|180519|0.1|0.1|0.0|0.0|0.1|0.2|0.3|
|**Order Item Id**|Order item code|180519|90260.0|52111.5|1.0|45130.5|90260.0|135389.5|180519.0|
|**Order Item Product Price**|Price of products without discount|180519|141.2|139.7|10.0|50.0|60.0|200.0|2000.0|
|**Order Item Profit Ratio**|Order Item Profit Ratio|180519|0.1|0.5|-2.8|0.1|0.3|0.4|0.5|
|**Order Item Quantity**|Number of products per order|180519|2.1|1.5|1.0|1.0|1.0|3.0|5.0|
|**Sales**|Value in sales|180519|203.8|132.3|10.0|120.0|199.9|300.0|2000.0|
|**Order Item Total**|Total amount per order|180519|183.1|120.0|7.5|104.4|164.0|247.4|1940.0|
|**Order Profit Per Order**|Order Profit Per Order|180519|22.0|104.4|-4275.0|7.0|31.5|64.8|911.8|
|**Order Zipcode**|Order Zip Code|24840|55426.1|31919.3|1040.0|23464.0|59405.0|90008.0|99301.0|
|**Product Card Id**|Product code|180519|692.5|336.4|19.0|403.0|627.0|1004.0|1363.0|
|**Product Category Id**|Product category code|180519|31.9|15.6|2.0|18.0|29.0|45.0|76.0|
|**Product Description**|Product Description|0|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|**Product Price**|Product Price|180519|141.2|139.7|10.0|50.0|60.0|200.0|2000.0|
|**Product Status**|Status of the product stock. If it is 1 not available, 0 the product is available|180519|0.0|0.0|0.0|0.0|0.0|0.0|0.0|

# **Introduction**
Supply chains are in simple terms; transporting goods from one place to another. It mostly starts from the production of the good and ends with handing over to the customer. 

“Supply chain management” as a term started to be used in the late 1980s and widely used in the 1990s. Before that period, most businesses preferred terms like “logistics” and “operations management” (Hugos, 2011). Today, “supply chain management” became the business standard as for the terminology. 

Supply chain management (SCM) has got increased importance and seen as one of the manufacturing approaches in the 21st century to be a more competitive organization. SCM has been accredited as a competitious method to integrate customers and suppliers by improved responsiveness and flexible manufacturing (Gunasekaran, 2004). Supply chains, once seen as product delivery, today represent a wider network where manufacturing, sales and delivery are carried out together. In this study, one of the primary objectives will be studied; the delivery of the products.

Timely deliveries has been the main target of supply chains since the term is being started to be used. Even more objectives are considered into while evaluating succesful supply chain management in recent times, timely and undamaged delivery is still in the the center of supply chains networks. 

With increasing use of internet and technlogy in supply chain management, data collecting, reaching and using the data became more accessible. As the data increased, machine learning and artificial intelligence techniques are worth more as the first choice of numerous businesses. 

Currently, the industries should improve their supply chains for reasons such as increasing marketshare, gaining consumers, generating the brand, and so on. There are many opportunities for the many sectors to capitalise. Machine learning, as one of the best options, can benefit the industries in tangible and intangible ways (Nagar et al., 2021). 
# **Related Work**
Machine learning techniques are used expecially in forecasting in supply chain management. Balancing between the supply and demand is one of the main objectives in many businesses. To be able to plan the supply, demand should be forecast accordingly. Machine learning models are found more favourable in demand forcasting than simpler forecasting methods (Carbonneau et al., 2008). Forecasting the demand is important due to planing and the scheduling manifacturing and delivery, two of the main branches of the supply chain management.

Study has been conducted with DataCo Supply Chain Dataset (DT Wiyanti et al., 2021) for demand forecasting and Deep Learning models are suggested to be used to achieve faster results and close outcomes with machine learning models. 

In our digital world, data-driven decision-making is necessary for any business as well as it is more accessible and results better outcomes. In supply chains, all plans are related with the correctly timing of moving goods. Delivering products early may overload warehouses and it costs more to keep excess goods. Besides that, delivering goods late will also influence businesses negatively. Predicting and providing just-in-time delivery is a key factor in the supply chain management. In my study, I will use 4 classification models (K-Nearest Neighborhood, Logistic Regression, Gaussian Naïve Bayes and Random Forest Classification) to predict and evaluate the late deliveries of orders in the DataCo Global Supply Chain Dataset.

Supervized machine learning models can be used with simulation models to forecast reliable delivery and help selecting resilient suppliers (Cavalcante et al., 2019). 

In my study, 2 regression models (linear regression model and decision tree regressor) also will be used to predict and evaluate the profitability of orders before the order is completed. Any positive outcome from a such study will ease making a data-driven business decisions as the profitability is the one of the main targets of most businesses.

Numerical and categorical features in the dataset will be considered seperately and together or prediction purposes in machine learning modelling and results will be compared with one another. Choosing the useful and effective features in machine learning models are helpful to improve the accuracy of machine learning models. Besides that, to be able to reduce the size of data without affecting the accuracy negatively will decrease time to run the models, and saves computing power which might be the limiting factor when running machine learning models.
# **Methodology**
CRISP-DM (CRoss-Industry Standard Process for Data Mining) methodology is used in this project. CRISP-DM is created to be industry neutral and used freely by wide range of data practitioners who are interested in data mining (Chapman et al., 2000).

This methodology (Figure 1.) consists of total 6 parts:

- Business Understanding: This is where one evaluate the requirement of business and create research questions for the project and planning the project overall.
- Data Understanding: This is the phase of collecting, describing and exploring of the business data.
- Data Preparation: In data preparation phase the data should be cleaned and reformatted as necessaty, as well as feature engineering for the modelling.
- Modeling: This is where we choose our machine learning models, then fit and test our experimental design.
- Evaluation: In evaluation phase we review the results and evaluate the outcomes.
- Deployment: This is the phase of model deployment and reporting our results with stakeholders.

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/CRISP-DM.png" alt="drawing" height="500"/>

### *Figure 1. CRISP-DM Methodology*

# **Data Preprocessing**

Cleaning the data for the modelling process has the foremost importance in order to achieve creditable conclusions. I checked all the attributes in my dataset for any attribute has less than 2 unique values (Table 3). 
### *Table 3. Unique Values of Attributes*

|**Attribute**|**Unique Values**|
| :- | :- |
|**Product Description**|0|
|**Product Status**|1|
|**Customer Password**|1|
|**Customer Email**|1|
|**Late\_delivery\_risk**|2|

All attributes that have less than 2 unique values can be dropped from the dataset as it offers no information for our models.

Next, I wanted to be sure the remaining attributes doesn’t have low variance, which may also not be valuable for future analysis (Table 4). One of the attributes (Order Item Discount Rate) has very low variance and required further analysis.
### *Table 4. Attributes with Low Variance*

|**Attribute**|**Variance**|
| :- | :- |
|**Order Item Discount Rate**|0.004958303|
|**Order Item Profit Ratio**|0.217898136|
|**Late\_delivery\_risk**|0.247669321|
|**Days for shipment (scheduled)**|1.889110823|
|**Order Item Quantity**|2.112521209|

After checking the values and value counts of the low variance variable (Table 5), it is the small values of the attribute resulting the low variance and the attribute should be kept in the dataset.
### *Table 5. Value Counts of Low Variance (Order Item Discount Rate) Attribute*

|**Value**|**Value counts**|
| :- | :- |
|**0.04**|10029|
|**0.15**|10029|
|**0.25**|10029|
|**0.20**|10029|
|**0.18**|10029|

There is no duplicated observation found In our dataset. However, there are null values in the dataset. Attributes that has more null values than more non-null values are simply dropped from the dataset. Null values in numerical attibutes filled with the mean value of the corresponding feature. Null values in categorical attributes are just filled with an empty string.

I checked variance to see whether any attribute provides information or not at its own. 

Then I need to see the correlation between attributes to check whether different attributes give the same or similar information in the dataset.

![](https://github.com/ktlszn/dataco-supply_chain/blob/main/images/Correlation%20Heatmap.png)
### *Figure 2. Correlation Heatmap of Attributes*

There are some highly correlated attributes in our dataset (Table 6.) and one of the highly correlated sets should be removed from the dataset. I limited my correlation coefficient limit for dropping attribute in 0.85 and up.
### *Table 6. Highly Correlated Numerical Attributes* 

|**Attribute1**|**Attribute2**|**Correlation**|
| :- | :- | :- |
|**Product Price**|**Order Item Product Price**|1|
|**Order Customer Id**|**Customer Id**|1|
|**Order Item Total**|**Sales per customer**|1|
|**Order Profit Per Order**|**Benefit per order**|1|
|**Order Item Total**|**Sales**|0.989744|
|**Sales**|**Sales per customer**|0.989744|
|**Order Profit Per Order**|**Order Item Profit Ratio**|0.823689|
|**Order Item Profit Ratio**|**Benefit per order**|0.823689|
|**Product Price**|**Sales**|0.789948|
|**Sales**|**Order Item Product Price**|0.789948|

# **Exploratory Data Analytics**
Standard class shipping is the highest share among all shipping modes (Figure 3.) as expected covering more than half of the shipping. It is followed by second class, first class and same day shipping respectively.

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/shipping%20mode%20(2).png" alt="drawing" height="500"/>

### *Figure 3. Shipping Mode Distribution of Sales*

First class and same day shipping modes are offering more luxury delivery options. However, proportion of late delivery among those categories are higher than standard class delivery (Figure 4).  

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/late%20delivery%20risk%20-%20shipping%20mode.png" alt="drawing" height="450"/>

### *Figure 4. Late Delivery Risk of Goods by Shipping Mode*

Monthly profit from the sales is relatively consistent through the years. However, the profit starts to drop at the end of 2017 (Figure 5.).

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/profit%20per%20month.png" alt="drawing" height="450"/>

### *Figure 5. Profit per Month between 2015-2018*

Looking at profit data together with sales data (Figure 6.) clarifies the sudden drop in profits on the late 2017. Declining sales clearly affected the monthly profit amount.

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/sales%20vs%20profit.png" alt="drawing" height="450"/>

### *Figure 6. Sales vs Profit*

Distribution of numerical attributes as also checked (Figure 7.) in case of any additional data cleaning might be required, and data with outliers are belong to profit of the orders.

![](https://github.com/ktlszn/dataco-supply_chain/blob/main/images/Numerical%20Attributes%20Distribution%20Boxplots.png)
### *Figure 7. Numerical Attributes Distribution*

The profit estimation is one of our targets in this study. Therefore, we will not remove outliers from the dataset.
# **Experimental Design**
Before the experimental design, features with datetime data type should be encoded into numerical data type. We have 2 datetime type features (shipping date (DateOrders), and order date (DateOrders)). 8 new numerical features (year, month, day, hour) are created from each of these datetime features.

In this project I will model and predict the late delivery risk of the orders in the supply chain data. 

First I need to check my target variable (Late\_delivery\_risk) for the imbalance. Running machine learning models in imbalanced data will induce misleading predictions.
### *Table 7. Target Attribute (Late\_delivery\_risk) Value Count*

|**Level**|**Value Count**|
| :- | :- |
|**1**|98977|
|**0**|81542|

Our target variable is aggreeably balanced, and no further action is required to correct it (Table 7.).
## **Machine Learning Validation Methods**
Validation in machine learning is the evaluation process of the trained model by using the testing dataset. I used 4 validation techiques to evaulate my models. Cross validation is used to prevent any bias that may arise from random sampling of train-test split by using multiple folds (groups of samples).
### **Train test Split (Hold-out Validation)**
Train test split (also named hold-out validation) is basically splitting dataset into train and test sets randomly. Train set is used for fitting the model and test set is used for validation.
### **Kfold**
Kfold divides the dataset into k sample groups (folds) and prediction function is learned by the k-1 folds and remaining fold is used for testing and validation (Scikit-learn, n.d.).	
### **Stratified Kfold**
Stratified kfold using the same method as kfold with only difference of having approximately same percentage of target variable in each fold.	
### **Shuffle**
Shuffle split model generates independent train test splits and uses train for fitting the model and test for validation for each split. 
### *Table 8. Classification Evaluation for Numerical Attributes (Late Delivery Prediction)*

|**Model**|**train\_test\_split**|**kfold\_5**|**strafifiedkfold\_5**|**shuffle**|
| :- | :- | :- | :- | :- |
|**KNN**|87.79|85.81|85.34|87.35|
|**Logistic Regression**|97.54|97.55|97.55|97.54|
|**Naive Bayes**|94.37|93.77|93.37|94.22|
|**Random Forest**|93.63|93.60|93.60|93.56|

Out of the 4 machine learning models, Logistic Regression gave the best result and more than 97% of the late deliveries in the test dataset is predicted accurately.

After predicting the late deliveries, I run the machine learning models for profit estimation.

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/Linear%20Regression%20Predictions.png" alt="drawing" height="350"/>

### *Figure 8. Linear Regression Model Predictions for Profit Estimation*

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/Decision%20Tree%20Regressor%20Predictions.png" alt="drawing" height="350"/>

### *Figure 9. Decision Tree Regressor Predictions for Profit Estimation*

### *Table 9. Regression Evaluation for Numerical Attributes (Profit Prediction)*

|**Model**|**r2\_score**|
| :- | :- |
|**Linear Regression**	|0.6893|
|**Decision Tree Regressor**|0.9483|

Out of the two regression models for profit estimation, Decision Tree Regressor has the best r2 score with 0.9482 or 94.82% (Table 9.). r2 score or r-squared is a statistical measure that gives the proportion of the variance for a target variable that can be explained by explanatory variable(s) in regression models.
## **Encoding Categorical Attributes**
Working with data in machine learning applications requires all data to be numerical. Therefore, we need to convert our categorical attributes to numeric ones to be able to use them in our models. 

The remaining 14 categorical features and their corresponding unique values are listed in Table 10 for encoding purposes.
### *Table 10. Unique Values of Categorical Attributes for Encoding Purposes*

|**Categorical Attribute**|**Unique Values**|
| :- | :- |
|**Customer Segment**|3|
|**Type**|4|
|**Delivery Status**|4|
|**Shipping Mode**|4|
|**Market**|5|
|**Order Status**|9|
|**Department Name**|11|
|**Order Region**|23|
|**Category Name**|50|
|**Product Name**|118|
|**Order Country**|164|
|**Customer City**|563|
|**Order State**|1089|
|**Order City**|3597|

I used two encoding methods for converting categorical attirbutes to numerical attributes. One hot encoding for attributes that has less than 8 unique values, and label encoding for attributes that has more than 8 unique values.
### **One Hot Encoding**
One hot encoding categorical features as one-hot numeric array. Basicly, it creates binary column for each category of the feature. It is most effective when the number of categories in a feature is few.
### **Label Encoding**
Label encoder encodes the target feature between 0 and number-of-classes minus 1. 

One more time I checked the highly correlated attributes as some categorical features are the just a function of categorical attributes and dropped one set of highly correlated pairs (correlation coefficient > 0.85) from my dataset. 
### *Table 11. Highly Correlated Encoded and Numerical Attributes*

|**Attribute1**|**Attribute2**|**Correlation**|
| :- | :- | :- |
|**Delivery Status\_Late delivery**|**Late\_delivery\_risk**|1|
|**order\_year**|**ship\_year**|0.994073297|
|**order\_month**|**ship\_month**|0.952178985|
|**Shipping Mode\_Standard Class**|**Days for shipment (scheduled)**|0.945696069|
|**ship\_year**|**Order Id**|0.942352822|
|**order\_year**|**Order Id**|0.941951847|
|**order\_hour**|**ship\_hour**|0.918932429|
|**Order Profit Per Order**|**Order Item Profit Ratio**|0.823689458|
|**Product Price**|**Order Item Total**|0.781781428|
|**Latitude**|**Customer Zipcode**|0.584550394|

After encoding, I rerun same machine learning models, without changing any parameter, and Logistic Regression is again the best model to predict the late deliveries (Table 12.).
### *Table 12. Classification Evaluation for All Attributes (Late Delivery Prediction)*

|**Model**|**train\_test\_split**|**kfold\_5**|**strafifiedkfold\_5**|**shuffle**|
| :- | :- | :- | :- | :- |
|**KNN**|86.28|85.81|82.24|85.89|
|**Logistic Regression**|97.51|97.55|97.36|97.56|
|**Naive Bayes**|79.32|93.77|79.50|79.54|
|**Random Forest**|96.62|93.60|95.90|95.71|

Accuracy is just over 97% with encoded data as well as running machine learning model with only numerical data.

### *Table 13. Classification Report for the Best Model (Logistic Regression)*

||**precision**|**recall**|**f1-score**|**support**|
| :- | :- | :- | :- | :- |
|**0**|0.982|0.962|0.972|20396|
|**1**|0.969|0.986|0.977|24734|
||
|**accuracy**||0.975|45130|
|**macro avg**|0.976|0.974|0.975|45130|
|**weighted avg**|0.975|0.975|0.975|45130|

Precision shows how much of our positive guesses were actually positive (for the value of 1). Recall gives how much positive we guessed correctly out of all positive results (for the value of 1). The f1-score gives the harmonic mean of precision and recall.

Precision and recall are 98.2% and %96.2 respectively for late delivery prediction (Table 13.). As our target variable is balanced, precision and recall for late delivery and on-time delivery are close to each other and the accuracy score of the model. 

<img src="https://github.com/ktlszn/dataco-supply_chain/blob/main/images/ROC_curve-confusion_matrix%20(4).png" alt="drawing" height="450"/>

### *Figure 10. Confusion Matrix and ROC Curve for the Best Model (Logistic Regression)*


ROC Curve shows the performance of a classification model at two classification thresholds; true positive rate and false positive rate. Area under the ROC Curve (AUC) gives the aggregate performance of the classification model. Our AUC for the logistic regression model covers 98% of the all performance evaluation area.

### *Table 14. Regression Evaluation for All Attributes (Profit Prediction)*

|**Model**|**r2\_score**|
| :- | :- |
|**Linear Regression**	|0.6921|
|**Decision Tree Regressor**|0.9475|

Regression models for profit estimation are also rerun after encoding the categorical features into numerical ones and similar r2 scores are resulted as running models with only numerical features.
# **Conclusion**
Delivery timing is one of the most important aspects in supply chain management. We are able to predict late delivery with 97% accuracy by using machine learning models. It is also clear that using only numerical data (sales, price and planned shipping date, etc.) gives the same accuracy with using all data we have by encoding the categorical data (customer segment, country, market data, etc.). We can also conclude that, by looking shipping mode and late delivery comparison, the late delivery risk for our project is an  overpromising problem rather than a planning issue.

It is also possible to predict the prospective profit from the orders before order is finalized with machine learning models. We can predict the profit with 5% margin of error when the order is placed. Using caterorical data along with numerical data gives the similar results for predicting the profit as using only the numerical data for our predictions. 


# References

Gunasekaran, A. (2004). Supply chain management: Theory and applications., 159(2), 265–268. doi:10.1016/j.ejor.2003.08.015

Carbonneau, R.; Laframboise, K.; & Vahidov, R. (2008). Application of machine learning techniques for supply chain demand forecasting., 184(3), 1140–1154. doi:10.1016/j.ejor.2006.12.004 

Hugos, M., (2011). Essentials of Supply Chain Management, NJ. John Wiley&Sons. International Energy Agency (IEA), <http://www.iea.org/>.

Nagar, D., Raghav, S., Bhardwaj, A., Kumar, R., Lata Singh, P., & Sindhwani, R. (2021). Machine learning: Best way to sustain the supply chain in the era of industry 4.0. Materials Today: Proceedings. doi:10.1016/j.matpr.2021.01.267

Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T.P., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide.

Cavalcante, I. M., Frazzon, E. M., Forcellini, F. A., & Ivanov, D. (2019). A supervised machine learning approach to data-driven simulation of resilient supplier selection in digital manufacturing. International Journal of Information Management, 49, 86–97. doi:10.1016/j.ijinfomgt.2019.03.004

DT Wiyanti et al 2021 J. Phys.: Conf. Ser. 1918 042012

Scikit-learn (n.d.) 3.1. Cross-validation: evaluating estimator performance.  Retrieved April 10, 2020, from https://scikit-learn.org/stable/modules/cross\_validation.html
