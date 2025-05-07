import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title='SUPPLY CHAIN DASHBOARD', page_icon='🚚', layout='wide')
st.markdown("""<style>[data-testid="stAppViewContainer"] {background: linear-gradient(45deg, #ff9a9e, #fad0c4, #fad0c4, #a18cd1, #fbc2eb);color: #333333;}</style>""", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #000000'>🚚 SUPPLY CHAIN DASHBOARD 🚚</h1>",unsafe_allow_html=True)
st.markdown("""<style>[data-testid='stSidebar']{background:linear-gradient(45deg, #ff9a9e, #fad0c4, #fad0c4, #a18cd1, #fbc2eb); color: #333333}</style>""",unsafe_allow_html=True)
st.markdown("""<style>[data-testid='stSidebar']{width:5px !important;}</style>""", unsafe_allow_html=True)
st.markdown("""<style>.stMetric{border-radius:15px;border:2px solid #4CAF50;padding:10px;}</style>""",unsafe_allow_html=True)
df=pd.read_csv("supply_chain_data.csv")
st.download_button(label="Get_Data", data=df.to_csv(index=False).encode('utf-8'), file_name="C:/Users/91808/Music/Nikita Documents/unified mentor internship/Supply Chain Management/supply_chain_data.csv",mime="csv")
Total_Revenue=df['Revenue generated'].sum()
Total_Products_Sold=df['Number of products sold'].sum()
Total_Cost=df['Costs'].sum()
Total_Stock_Levels=df['Stock levels'].sum()
Average_Lead_Times=df['Lead times'].mean()
Average_Shipping_Times=df['Shipping times'].mean()
col1,col2,col3=st.columns(3)
col1.metric('💸Total_Revenue',f'₹{Total_Revenue:,.0f}')
col2.metric('📦Total_Products_Sold',f'{Total_Products_Sold:,.0f}')
col3.metric('💰Total_Cost',f'₹{Total_Cost:,.0f}')
col4,col5,col6=st.columns(3)
col4.metric('📈Total_Stock_Levels',f'{Total_Stock_Levels:,.0f}')
col5.metric('⏱️Average_Lead_Times',f'{Average_Lead_Times}days')
col6.metric('🚚Average_Shipping_Times',f'{Average_Shipping_Times}days')

st.sidebar.header('🎯Filters')
Selected_Supplier=st.sidebar.multiselect('Select_Supplier',df['Supplier name'].unique())
Selected_Location=st.sidebar.multiselect('Select_Location',df['Location'].unique())
Selected_Product=st.sidebar.multiselect('Select_Product',df['Product type'].unique())
Selected_Transportation_Mode=st.sidebar.multiselect('Select_Transportation_Mode',df['Transportation modes'].unique)
filtered_df=df.copy()
if Selected_Supplier:
    filtered_df=filtered_df[filtered_df['Supplier name'].isin(Selected_Supplier)]
if Selected_Location:
    filtered_df=filtered_df[filtered_df['Location'].isin(Selected_Location)]
if Selected_Product:
    filtered_df=filtered_df[filtered_df['Product type'].isin(Selected_Product)]
if Selected_Transportation_Mode:
    filtered_df=filtered_df[filtered_df['Transportation modes'].isin(Selected_Transportation_Mode)]
st.markdown("---")    

col1,col2,col3=st.columns(3)
with col1:
#Chart1
    revenue_by_product=filtered_df.groupby('Product type')['Revenue generated'].sum().sort_values(ascending =False).reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax=sns.barplot(data=revenue_by_product,x='Product type',y='Revenue generated',palette='Blues_d',edgecolor='none')
    bars = ax.containers[0]
    ax.bar_label(bars,fmt='%.f',padding=3)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Product Type')
    plt.ylabel('Revenue By Product')
    plt.title('Top Performing Product Type', weight='bold',fontsize=12)
    st.pyplot(fig)
    
with col2:
#Chart2
    location_stats=filtered_df.groupby('Location')['Production volumes'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,5), facecolor='none')
    labels=location_stats['Location']
    sizes=location_stats['Production volumes']
    colors = ['lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'cornflowerblue', 'steelblue']
    plt.pie(sizes,labels=labels,autopct='%.2f%%',explode=[0]*len(labels),wedgeprops=dict(width=0.7),colors=colors[:len(sizes)])
    center_circle=plt.Circle((0,0),0.70,fc='none')
    fig.patch.set_edgecolor('black')  
    fig.patch.set_linewidth(1) 
    plt.gca().add_artist(center_circle)
    plt.title('Percentage of Production Volumes Aligned with Market Demands by Location',weight='bold',fontsize=12)
    st.pyplot(plt)   

with col3:
#Chart3    
    supplier_statistics=filtered_df.groupby('Supplier name')[['Manufacturing costs','Defect rates']].sum().reset_index()
    fig,ax1=plt.subplots(figsize=(8,5))
    ax1=sns.barplot(data=supplier_statistics,x='Supplier name',y='Manufacturing costs',label='Manufacturing costs',ax=ax1,palette='Greens_d')
    bars=ax1.containers[0]
    ax1.bar_label(bars,fmt='%.1f')
    ax2=ax1.twinx()
    sns.lineplot(data=supplier_statistics, x='Supplier name', y='Defect rates', ax=ax2, marker='o', color='steelblue')
    for i , value in enumerate(supplier_statistics['Defect rates']):
        ax2.text(supplier_statistics['Supplier name'][i],value+0.3,f'{value:.1f}',ha='center',color='black',weight='bold')
    ax1.set_xlabel('Supplier name')
    fig.patch.set_facecolor('none')
    ax1.set_facecolor('none')
    ax1.set_ylabel('Manufacturing cost')
    ax2.set_ylabel('Defect rates')
    plt.title('Relationship Between Defect Rates and Manufacturing Costs by Supplier',weight='bold',fontsize=12)
    st.pyplot(fig)
    
col1,col2,col3=st.columns(3)    
with col1: 
    product_statistics= filtered_df.groupby('Product type')['Stock levels'].mean().reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax=sns.barplot(data=product_statistics,x='Product type',y='Stock levels',palette='colorblind')
    ax.bar_label(ax.containers[0], fmt='%.0f')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.title('Impact of Stocks levels on products',weight='bold',fontsize=12)
    plt.xlabel('Product type')
    plt.ylabel('Avg of Stock levels')
    st.pyplot(fig)
    
with col2:
#Chart5
    location_revenue=filtered_df.groupby('Location')['Revenue generated'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,5), facecolor='none')
    labels=location_revenue['Location']
    sizes=location_revenue['Revenue generated']
    colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0]*len(labels),colors=colors[:len(sizes)])
    fig.patch.set_edgecolor('black')  
    fig.patch.set_linewidth(1) 
    plt.title('Revenue Distribution by Location',weight='bold',fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

with col3:
    #Chart6
    supplier_revenue=filtered_df.groupby('Supplier name')['Revenue generated'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,5), facecolor='none')
    labels=supplier_revenue['Supplier name']
    sizes=supplier_revenue['Revenue generated']
    colors=['lightgreen', 'mediumseagreen', 'seagreen', 'limegreen', 'forestgreen', 'darkgreen']
    plt.pie(sizes,labels=labels,autopct='%.1f%%',explode=[0.01]*len(labels),wedgeprops=dict(width=0.4),colors=colors[:len(sizes)])
    center_circle=plt.Circle((0,0),0.70,fc='none')
    fig.patch.set_edgecolor('black')  
    fig.patch.set_linewidth(1) 
    plt.gca().add_artist(center_circle)
    plt.title('Revenue Contribution by Supplier',weight='bold',fontsize=12)
    st.pyplot(plt)
    

col1,col2,col3=st.columns(3)
with col1:
    #Chart7
    leadtime=filtered_df.groupby('Supplier name')['Lead times'].mean().reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax=sns.lineplot(data=leadtime,x='Supplier name',y='Lead times',marker='>')
    for i , value in enumerate(leadtime['Lead times']):
        ax.text(leadtime['Supplier name'][i],value,f'{value:.1f}',color='black',weight='bold',fontsize=9)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.xlabel('Supplier name')
    plt.ylabel('Average lead time')
    plt.title('Lead Time by Supplier',weight='bold',fontsize=12)
    st.pyplot(fig)

with col2:
    #Chart8
    producttype_stat=filtered_df.groupby('Product type')[['Manufacturing costs','Price']].mean().reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    index=np.arange(len(producttype_stat))
    bar_width=0.34
    bars1=plt.bar(index,producttype_stat['Price'],bar_width,label='Price',color='Teal')
    bars2=plt.bar(index+bar_width,producttype_stat['Manufacturing costs'],bar_width,label='Manufacturing costs',color='Coral')
    plt.xticks(index +bar_width/2,producttype_stat.index, rotation=45)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.bar_label(bars1,fmt='%.2f')
    plt.bar_label(bars2,fmt='%.2f')
    plt.ylabel('Amount')
    plt.title('Comparison of price and manufacturing costs by product type',weight='bold',fontsize=12)
    plt.legend()
    st.pyplot(fig)    

with col3:
    #Chart9
    unit_sold_stat=filtered_df.groupby('Product type')['Number of products sold'].sum().reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax=sns.barplot(data=unit_sold_stat,x='Product type',y='Number of products sold',palette='Reds')
    bars=ax.containers[0]
    ax.bar_label(bars,fmt='%.f')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.xlabel('Product type')
    plt.ylabel('Number of products sold')
    plt.title('Units Sold by Product Type',weight='bold',fontsize=12)
    st.pyplot(fig)
    
col1,col2,col3=st.columns(3)
with col1:
    #Chart10
    transportation_stats=filtered_df.groupby('Transportation modes')['Shipping costs'].mean().reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    ax=sns.barplot(data=transportation_stats,y='Transportation modes',x='Shipping costs',palette='viridis')
    bars=ax.containers[0] 
    ax.bar_label(bars,fmt='%.f')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.xlabel('Shipping costs')
    plt.ylabel('Transportation modes')
    plt.title('Shipping Cost by Transportation Mode',weight='bold',fontsize=12)
    st.pyplot(fig)

with col2:
    #Chart11
    location_revenue=filtered_df.groupby('Location')['Revenue generated'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,5), facecolor='none')
    labels=location_revenue['Location']
    sizes=location_revenue['Revenue generated']
    colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0]*len(labels),colors=colors[:len(sizes)])
    fig.patch.set_edgecolor('black')  
    fig.patch.set_linewidth(1) 
    plt.title('Revenue Distribution by Location',weight='bold',fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

with col3:
    #Chart12
    carrier_statistics=filtered_df.groupby('Shipping carriers')[['Shipping times','Shipping costs']].mean().reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    bar_width=0.34
    index=np.arange(len(carrier_statistics['Shipping carriers']))
    bars1=plt.bar(index,carrier_statistics['Shipping times'],bar_width,label='Shipping times',color='steelblue')
    bars2=plt.bar(index+bar_width,carrier_statistics['Shipping costs'],bar_width,label='Shipping costs',color='lightgreen')
    plt.xticks(index+bar_width/2,carrier_statistics.index,rotation=45)
    plt.bar_label(bars1,fmt='%.2f')
    plt.bar_label(bars2,fmt='%.2f')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.xlabel('Shipping Carriers')
    plt.title('Shipping Carrier Performance',weight='bold',fontsize=12)
    plt.legend()
    st.pyplot(fig)
    

col1,col2,col3,col4=st.columns(4)
with col1:
    #Chart13
    transportation_stats=filtered_df.groupby('Transportation modes')['Order quantities'].sum().reset_index()
    plt.figure(figsize=(8,5),facecolor='none')
    sns.heatmap(transportation_stats['Order quantities'].values.reshape(-1,1),annot=True,fmt='.0f',cmap='YlGnBu',
            yticklabels=transportation_stats['Transportation modes'],xticklabels=['Order quantities'])
    plt.xlabel('')
    plt.ylabel('Transportation modes')
    plt.title('Total Order Quantity by Transportation Mode',weight='bold',fontsize=12)
    st.pyplot(plt)

with col2:
    #Chart14
    fig,ax=plt.subplots(figsize=(8,5))
    sns.lineplot(data=filtered_df,x='Price',y='Revenue generated',marker='o',color='blue')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.title('Revenue Generated by Price Range',weight='bold',fontsize=12)
    plt.xlabel('Price')
    plt.ylabel('Revenue Generated')
    st.pyplot(fig)

with col3:
    #Chart15
    locationstats=filtered_df.groupby('Location')['Order quantities'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax=sns.barplot(data=locationstats,x='Location',y='Order quantities',palette='muted')
    bars=ax.containers[0]
    ax.bar_label(bars,fmt='%.f')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.xlabel('Location')
    plt.ylabel('Order Quantities')
    plt.title('Order Quantities by Location',weight='bold',fontsize=12)
    st.pyplot(fig)
    
with col4:
    #Chart16
    filtered_df['Profit']=filtered_df['Revenue generated']-filtered_df['Manufacturing costs']
    profitby_prod=filtered_df.groupby('Product type')['Profit'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax=sns.barplot(data=profitby_prod,x='Product type',y='Profit',palette='Paired')
    bars=ax.containers[0]
    ax.bar_label(bars,fmt='%.f')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.xlabel('Product type')
    plt.ylabel('Profit')
    plt.title('Overall Profitability by Product Type',weight='bold',fontsize=12)
    st.pyplot(fig)

filtered_df['Sales_per_stock_unit']=filtered_df['Revenue generated']/filtered_df['Stock levels']
filtered_df['Profit_per_product']=filtered_df['Revenue generated']-filtered_df['Manufacturing costs']
filtered_df['Defect_percentage']=filtered_df['Defect rates']*100
filtered_df['Shipping_cost_per_product']=filtered_df['Shipping costs']/filtered_df['Number of products sold']
filtered_df['Order_to_production_ratio']=filtered_df['Order quantities']/filtered_df['Production volumes']
filtered_df['Total_lead_time']=filtered_df['Lead time']+filtered_df['Manufacturing lead time']

categorical_columns=['Product type','Availability','Shipping carriers','Inspection results','Transportation modes', 'Routes','Customer demographics','Supplier name', 'Location']

filtered_df_encoded=pd.get_dummies(filtered_df,columns=categorical_columns)
drop_col=['SKU']

filtered_df_model=filtered_df_encoded.drop(columns=drop_col)
    
from sklearn.model_selection import train_test_split
X=filtered_df_model.drop(columns=['Revenue generated'])
y=filtered_df_model['Revenue generated']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
st.title("Supply Chain Model Results")
print('MSE:',mse)
print('R2:',r2)
print('MAE:',mae)
print('RMSE:',rmse)
st.metric("MSE", f"{mse:.2f}")
st.metric("R2", f"{r2:.2f}")
st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmae:.2f}")

fig,ax=plt.subplots(figsize=(8,5))
residuals=y_test-y_pred
plt.figure(figsize=(10,6))
sns.histplot(residuals,kde=True)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title("Residuals Distribution")
st.pyplot(fig)

fig,ax=plt.subplots(figsize=(8,5))
plt.figure(figsize=(10,6))
plt.scatter(y_pred,residuals)
plt.axhline(y=0,color='r',linestyle='--')
plt.title('Residual Vs Predicted Plot')
plt.xlabel('Predicted Revenue')
plt.ylabel('Residuals')
st.pyplot(fig)

coefficient=model.coef_
features=X.columns
coeff_df=pd.DataFrame({'Features':features,'Coefficient':coefficient})
print(coeff_df.sort_values(by='Coefficient',ascending=False))
fig,ax=plt.subplots(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs Predicted Revenue')
plt.legend()
st.pyplot(fig)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model=keras.Sequential([layers.Dense(128,activation='relu'),
                       layers.Dense(64,activation="relu"),
                        layers.Dense(32,activation='relu'),
                        layers.Dense(3)])
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
history=model.fit(X_train_scaled,y_train,epochs=50,batch_size=32,validation_split=0.2)

fig,ax=plt.subplots(figsize=(8,5))
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss Curve')
plt.legend()
st.pyplot(fig)

test_loss,test_mae=model.evaluate(X_test_scaled,y_test)
print(f"Test MAE:{test_mae}")
y_predict=model.predict(X_test_scaled)
y_predict_df=pd.DataFrame(y_predict,columns=['Predicted_Revenue','Predicted_Profit','Predicted_No._of_Unit_Sold'])
y_test_df=y_test.reset_index(drop=True)
result_df=pd.concat([y_test_df,y_predict_df],axis=1)

st.metric("Test MAE", f"{test_mae:.2f}")
st.line_chart(history.history['loss'], use_container_width=True)
st.dataframe(results_df.head())

st.markdown("<p style='text-align: center; font-size: 12px;'>Made by Nikita Mendhe | <a href='www.linkedin.com/in/nikita-mendhe-2067b5210' target='_blank'>LinkedIn</a></p>", unsafe_allow_html=True)
