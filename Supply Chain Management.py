import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("supply_chain_data.csv")
st.set_page_config(page_title='SUPPLY CHAIN DASHBOARD', page_icon='🚚', layout='wide')
st.markdown("""<style>.block-container { padding-top: 0rem; }</style>""", unsafe_allow_html=True)
st.markdown("""<style>[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #a18cd1, #000046);color: #ffffff;
div[data-testid='stMultiSelect']{width:100px !important; font-size:10px !importnat; padding:1px !importnat;}label{font-size:10px !important; color:white !important;}</style>""", unsafe_allow_html=True)
filtered_df=df.copy() 
col1,col2,col3,col4,col5=st.columns([4,0.7,0.7,0.7,1])
col1.markdown("<h1 style='text-align: left; color: #ffffff;font-size: 45px; font-weight: bold;'>🚚 SUPPLY CHAIN DASHBOARD </h1>",unsafe_allow_html=True)
Selected_Supplier=col2.multiselect('Select_Supplier',df['Supplier name'].unique())
Selected_Location=col3.multiselect('Select_Location',df['Location'].unique())
Selected_Product=col4.multiselect('Select_Product',df['Product type'].unique())
Selected_Transportation_Mode=col5.multiselect('Select_Mode',df['Transportation modes'].unique())
if Selected_Supplier:
    filtered_df=filtered_df[filtered_df['Supplier name'].isin(Selected_Supplier)]
if Selected_Location:
    filtered_df=filtered_df[filtered_df['Location'].isin(Selected_Location)]
if Selected_Product:
    filtered_df=filtered_df[filtered_df['Product type'].isin(Selected_Product)]
if Selected_Transportation_Mode:
    filtered_df=filtered_df[filtered_df['Transportation modes'].isin(Selected_Transportation_Mode)]
st.markdown("""<style>div[data-testid="stDownloadButton"] button { background-color: #007bff !important; border: 2px solid white !important; color: white !important; /
padding: 10px 20px !important; border-radius: 12px !important; font-size: 16px !important;font-weight: bold !important;} </style>""", unsafe_allow_html=True)
st.download_button(label="Dataset" ,data=df.to_csv(index=False).encode('utf-8'), file_name="supply_chain_data.csv",mime="csv")
Total_Revenue=df['Revenue generated'].sum()
Total_Products_Sold=df['Number of products sold'].sum()
Total_Cost=df['Costs'].sum()
Total_Stock_Levels=df['Stock levels'].sum()
Average_Lead_Times=df['Lead times'].mean()
Average_Shipping_Times=df['Shipping times'].mean()
st.markdown("""<style>.metric-container{ text-align: center;color: white !important;font-size:24px;font-weight: bold;} .metric-value{text-align: center;fontsize:24px;font-weight: bold; display:block; margin-top:5px} </style>""", unsafe_allow_html=True)
col1,col2,col3,col4,col5,col6=st.columns(6)
col1.markdown(f'<div class="metric-text">💸Total_Revenue<span class="metric-value">₹{Total_Revenue:,.0f}</span></div>',unsafe_allow_html=True)
col2.markdown(f'<div class="metric-text">📦Total_Products_Sold<span class="metric-value">{Total_Products_Sold:,.0f}</span></div>',unsafe_allow_html=True)
col3.markdown(f'<div class="metric-text">💰Total_Cost<span class="metric-value">₹{Total_Cost:,.0f}</span></p>',unsafe_allow_html=True)
col4.markdown(f'<div class="metric-text">📈Total_Stock_Levels<span class="metric-value">{Total_Stock_Levels:,.0f}</span></div>',unsafe_allow_html=True)
col5.markdown(f'<div class="metric-text">⏱️Average_Lead_Times<span class="metric-value">{Average_Lead_Times}days </span></div>',unsafe_allow_html=True)
col6.markdown(f'<div class="metric-text">🚚Average_Shipping_Times<span class="metric-value">{Average_Shipping_Times}days </span></div>',unsafe_allow_html=True)
with st.container():
    st.markdown("<h2 style='color:white;'>🔹 Sales & Product Performance</h2>", unsafe_allow_html=True)   
    col1,col2,col3=st.columns(3)
    with col1:
    #Chart1
        revenue_by_product=filtered_df.groupby('Product type')['Revenue generated'].sum().sort_values(ascending =False).reset_index()
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        ax=sns.barplot(data=revenue_by_product,x='Product type',y='Revenue generated',palette='Blues_d',edgecolor='none')
        bars = ax.containers[0]
        ax.bar_label(bars,fmt='%.f',padding=3,color='white')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Product Type',color='white')
        plt.ylabel('Revenue By Product',color='white')
        plt.title('Top Performing Product Type', weight='bold',fontsize=12,color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with col2:
    #Chart2
        location_revenue=filtered_df.groupby('Location')['Revenue generated'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(4.5,3.5), facecolor='none')
        labels=location_revenue['Location']
        sizes=location_revenue['Revenue generated']
        colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        ax.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0]*len(labels),colors=colors[:len(sizes)],textprops={'color': 'white'})
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth(1) 
        ax.title('Revenue Distribution by Location',weight='bold',fontsize=12,color='white')
        ax.xticks(color='white') 
        ax.yticks(color='white')
        ax.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
    with col3:
    #chart 3  
        supplier_revenue=filtered_df.groupby('Supplier name')['Revenue generated'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(4.5,3.5), facecolor='none')
        labels=supplier_revenue['Supplier name']
        sizes=supplier_revenue['Revenue generated']
        colors=['lightgreen', 'mediumseagreen', 'seagreen', 'limegreen', 'forestgreen', 'darkgreen']
        ax.pie(sizes,labels=labels,autopct='%.1f%%',explode=[0.01]*len(labels),wedgeprops=dict(width=0.4),colors=colors[:len(sizes)],textprops={'color': 'white'})
        center_circle=plt.Circle((0,0),0.70,fc='none')
        fig.patch.set_edgecolor('black')  
        fig.patch.set_linewidth(1) 
        ax.gca().add_artist(center_circle)
        ax.title('Revenue Contribution by Supplier',weight='bold',fontsize=12,color='white')
        ax.xticks(color='white') 
        ax.yticks(color='white')
        ax.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
with st.container():    
    col1,col2,col3=st.columns(3)  
    with col1:
        unit_sold_stat=filtered_df.groupby('Product type')['Number of products sold'].sum().reset_index()
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        ax=sns.barplot(data=unit_sold_stat,x='Product type',y='Number of products sold',palette='Reds')
        bars=ax.containers[0]
        ax.bar_label(bars,fmt='%.f',color='white')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xlabel('Product type',color='white')
        plt.ylabel('Number of products sold',color='white')
        plt.title('Units Sold by Product Type',weight='bold',fontsize=12,color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        sns.lineplot(data=filtered_df,x='Price',y='Revenue generated',marker='o',color='red')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.title('Revenue Generated by Price Range',weight='bold',fontsize=12,color='white')
        plt.xlabel('Price',color='white')
        plt.ylabel('Revenue Generated',color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
    with col3:   
        filtered_df['Profit']=filtered_df['Revenue generated']-filtered_df['Manufacturing costs']
        profitby_prod=filtered_df.groupby('Product type')['Profit'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        ax=sns.barplot(data=profitby_prod,x='Product type',y='Profit',palette='Paired')
        bars=ax.containers[0]
        ax.bar_label(bars,fmt='%.f',color='white')
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        plt.xlabel('Product type',color='white')
        plt.ylabel('Profit',color='white')
        plt.title('Overall Profitability by Product Type',weight='bold',fontsize=12,color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
with st.container():
    st.markdown("<h2 style='color:white;'>🔹 Inventory & Stock Insights</h2>", unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        product_statistics= filtered_df.groupby('Product type')['Stock levels'].mean().reset_index()
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        ax=sns.barplot(data=product_statistics,x='Product type',y='Stock levels',palette='colorblind')
        ax.bar_label(ax.containers[0], fmt='%.0f',color='white')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.title('Impact of Stocks levels on products',weight='bold',fontsize=12,color='white')
        plt.xlabel('Product type',color='white')
        plt.ylabel('Avg of Stock levels',color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with col2:
        locationstats=filtered_df.groupby('Location')['Order quantities'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        ax=sns.barplot(data=locationstats,x='Location',y='Order quantities',palette='muted')
        bars=ax.containers[0]
        ax.bar_label(bars,fmt='%.f',color='white')
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        plt.xlabel('Location',color='white')
        plt.ylabel('Order Quantities',color='white')
        plt.title('Order Quantities by Location',weight='bold',fontsize=12,color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with col3:
        fig_hm, ax_hm = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(transportation_stats['Order quantities'].values.reshape(-1, 1),annot=True, fmt='.0f', cmap='YlGnBu',yticklabels=transportation_stats['Transportation modes'],xticklabels=['Order quantities'], ax=ax_hm)
        ax_hm.set_xlabel('')
        ax_hm.set_ylabel('Transportation modes', color='white')
        ax_hm.set_title('Total Order Quantity by Transportation Mode', weight='bold', fontsize=12, color='white')
        fig_hm.patch.set_facecolor('none')
        ax_hm.set_facecolor('none')
        plt.tight_layout()
        st.pyplot(fig_hm, use_container_width=True)
        
with st.container():
    st.markdown("<h2 style='color:white;'>🔹 Cost, Pricing & Defects</h2>", unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        supplier_statistics=filtered_df.groupby('Supplier name')[['Manufacturing costs','Defect rates']].sum().reset_index()
        fig,ax1=plt.subplots(figsize=(4.5,3.5))
        ax1=sns.barplot(data=supplier_statistics,x='Supplier name',y='Manufacturing costs',label='Manufacturing costs',ax=ax1,palette='Greens_d')
        bars=ax1.containers[0]
        ax1.bar_label(bars,fmt='%.1f',color='white')
        ax2=ax1.twinx()
        sns.lineplot(data=supplier_statistics, x='Supplier name', y='Defect rates', ax=ax2, marker='o', color='steelblue')
        ax2.set_ylabel('Defect Rates', color='white')
        ax2.tick_params(axis='y', color='white')
        ax1.set_xlabel('Supplier name',color='white')
        fig.patch.set_facecolor('none')
        ax1.set_facecolor('none')
        ax1.set_ylabel('Manufacturing cost',color='white')
        ax2.set_ylabel('Defect rates',color='white')
        plt.title('Relationship Between Defect Rates and Manufacturing Costs by Supplier',weight='bold',fontsize=12,color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with col2:
        producttype_stat=filtered_df.groupby('Product type')[['Manufacturing costs','Price']].mean().reset_index()
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        index=np.arange(len(producttype_stat))
        bar_width=0.34
        bars1=plt.bar(index,producttype_stat['Price'],bar_width,label='Price',color='Teal')
        bars2=plt.bar(index+bar_width,producttype_stat['Manufacturing costs'],bar_width,label='Manufacturing costs',color='Coral')
        plt.xticks(index +bar_width/2,producttype_stat.index, rotation=45)
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.bar_label(bars1,fmt='%.2f',color='white')
        plt.bar_label(bars2,fmt='%.2f',color='white')
        plt.ylabel('Amount',color='white')
        plt.title('Comparison of price and manufacturing costs by product type',weight='bold',fontsize=12,color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig, use_container_width=True)
    with col3:
        inspection_results_stats=filtered_df.groupby('Inspection results')['Defect rates'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(4.5,3.5), facecolor='none')
        labels=inspection_results_stats['Inspection results']
        sizes=inspection_results_stats['Defect rates']
        plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0]*len(labels),textprops={'color': 'white'})
        fig.patch.set_edgecolor('black')  
        fig.patch.set_linewidth(1) 
        plt.title('Defect Rates by Inspection Results',weight='bold',fontsize=12,color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

with st.container():
    st.markdown("<h2 style='color:white;'>🔹 Shipping & Supply Chain Performance</h2>", unsafe_allow_html=True)
    col1,col2,col3,col4=st.columns(4)
    with col1:
        transportation_stats=filtered_df.groupby('Transportation modes')['Shipping costs'].mean().reset_index()
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        ax=sns.barplot(data=transportation_stats,y='Transportation modes',x='Shipping costs',palette='viridis')
        bars=ax.containers[0] 
        ax.bar_label(bars,fmt='%.f')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xlabel('Shipping costs',color='white')
        plt.ylabel('Transportation modes',color='white')
        plt.title('Shipping Cost by Transportation Mode',weight='bold',fontsize=12,color='white')
        plt.tight_layout()
        plt.xticks(color='white') 
        plt.yticks(color='white')
        st.pyplot(fig, use_container_width=True)
    with col2:    
        carrier_statistics=filtered_df.groupby('Shipping carriers')[['Shipping times','Shipping costs']].mean().reset_index()
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        bar_width=0.34
        index=np.arange(len(carrier_statistics['Shipping carriers']))
        bars1=plt.bar(index,carrier_statistics['Shipping times'],bar_width,label='Shipping times',color='steelblue')
        bars2=plt.bar(index+bar_width,carrier_statistics['Shipping costs'],bar_width,label='Shipping costs',color='lightgreen')
        plt.xticks(index+bar_width/2,carrier_statistics.index,rotation=45)
        plt.bar_label(bars1,fmt='%.2f',color='white')
        plt.bar_label(bars2,fmt='%.2f',color='white')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xlabel('Shipping Carriers',color='white')
        plt.title('Shipping Carrier Performance',weight='bold',fontsize=12,color='white')
        plt.tight_layout()
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.legend()
        st.pyplot(fig, use_container_width=True)
    with col3:
        leadtime=filtered_df.groupby('Supplier name')['Lead times'].mean().reset_index()
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        ax=sns.lineplot(data=leadtime,x='Supplier name',y='Lead times',marker='>')
        for i , value in enumerate(leadtime['Lead times']):
            ax.text(leadtime['Supplier name'][i],value,f'{value:.1f}',color='white',weight='bold',fontsize=9)
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xlabel('Supplier name',color='white')
        plt.ylabel('Average lead time',color='white')
        plt.title('Lead Time by Supplier',weight='bold',fontsize=12,color='white')
        plt.tight_layout()
        plt.xticks(color='white') 
        plt.yticks(color='white')
        st.pyplot(fig, use_container_width=True)
    
#feature engineering
filtered_df['Sales_per_stock_unit']=filtered_df['Revenue generated']/filtered_df['Stock levels']
filtered_df['Profit_per_product']=filtered_df['Revenue generated']-filtered_df['Manufacturing costs']
filtered_df['Defect_percentage']=filtered_df['Defect rates']*100
filtered_df['Shipping_cost_per_product']=filtered_df['Shipping costs']/filtered_df['Number of products sold']
filtered_df['Order_to_production_ratio']=filtered_df['Order quantities']/filtered_df['Production volumes']
filtered_df['Total_lead_time']=filtered_df['Lead time']+filtered_df['Manufacturing lead time']
categorical_columns=['Product type','Availability','Shipping carriers','Inspection results','Transportation modes', 'Routes','Customer demographics','Supplier name', 'Location']
filtered_df_encoded=pd.get_dummies(filtered_df,columns=categorical_columns)
drop_col=['SKU']
#modeling
filtered_df_model=filtered_df_encoded.drop(columns=drop_col)
#modeling and forecasting   
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
print('MSE:',mse)
print('R2:',r2)
print('MAE:',mae)
print('RMSE:',rmse)
residuals=y_test-y_pred
with st.container():
    st.markdown("<h2 style='color:white;'>🔹 Model Performance Visuals</h2>", unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        ax.set_xlabel('Residual',color='white')
        ax.set_ylabel('Frequency',color='white')
        ax.set_title("Residuals Distribution",color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        ax.scatter(y_pred, residuals)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        ax.set_title('Residual Vs Predicted Plot',color='white')
        ax.set_xlabel('Predicted Revenue',color='white')
        ax.set_ylabel('Residuals',color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
with st.container():   
    col1,col2=st.columns(2)
    with col1:
        coefficient=model.coef_
        features=X.columns
        coeff_df=pd.DataFrame({'Features':features,'Coefficient':coefficient})
        print(coeff_df.sort_values(by='Coefficient',ascending=False))
        fig,ax=plt.subplots(figsize=(8,5))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xlabel('Actual Revenue',color='white')
        plt.ylabel('Predicted Revenue',color='white')
        plt.xticks(color='white') 
        plt.yticks(color='white')
        plt.title('Actual vs Predicted Revenue',color='white')
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig, use_container_width=True)

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
    with col2:
        fig,ax=plt.subplots(figsize=(4.5,3.5))
        plt.plot(history.history['loss'],label='Training Loss')
        plt.plot(history.history['val_loss'],label='Validation Loss')
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Model Loss Curve')
        plt.xticks(color='white') 
        plt.yticks(color='white') 
        plt.tight_layout()
        plt.legend()
        st.pyplot(fig, use_container_width=True)
st.markdown(""" <style> .metric-label, .metric-value {color: white !important;}.stDataFrame, .stDataFrame div {color: white !important;}.stDataFrame table tbody tr td {color: white !important;}</style>""", unsafe_allow_html=True)
test_loss,test_mae=model.evaluate(X_test_scaled,y_test)
print(f"Test MAE:{test_mae}")
y_predict=model.predict(X_test_scaled)
y_predict_df=pd.DataFrame(y_predict,columns=['Predicted_Revenue','Predicted_Profit','Predicted_No._of_Unit_Sold'])
y_test_df=y_test.reset_index(drop=True)
result_df=pd.concat([y_test_df,y_predict_df],axis=1)
st.dataframe(result_df.head())
st.markdown("<p style='text-align: center; font-size: 12px;'>Made by Nikita Mendhe | <a href='www.linkedin.com/in/nikita-mendhe-2067b5210' target='_blank'>LinkedIn</a></p>", unsafe_allow_html=True)
