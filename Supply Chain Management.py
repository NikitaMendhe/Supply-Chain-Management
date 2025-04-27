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

col1,col2=st.columns(2)
with col1:
#Chart1
    revenue_by_product=filtered_df.groupby('Product type')['Revenue generated'].sum().sort_values(ascending =False).reset_index()
    fig,ax=plt.subplots(figsize=(12,8))
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
    fig, ax = plt.subplots(figsize=(10,6), facecolor='none')
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
    
col1,col2=st.columns(2)
with col1:
#Chart3    
    supplier_statistics=filtered_df.groupby('Supplier name')[['Manufacturing costs','Defect rates']].sum().reset_index()
    fig,ax1=plt.subplots(figsize=(12,8))
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
    
with col2: 
    product_statistics= filtered_df.groupby('Product type')['Stock levels'].mean().reset_index()
    fig,ax=plt.subplots(figsize=(12,8))
    ax=sns.barplot(data=product_statistics,x='Product type',y='Stock levels',palette='colorblind')
    ax.bar_label(ax.containers[0], fmt='%.0f')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.title('Impact of Stocks levels on products',weight='bold',fontsize=12)
    plt.xlabel('Product type')
    plt.ylabel('Avg of Stock levels')
    st.pyplot(fig)
    
col1,col2=st.columns(2)
with col1:
#Chart5
    location_revenue=filtered_df.groupby('Location')['Revenue generated'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10,6), facecolor='none')
    labels=location_revenue['Location']
    sizes=location_revenue['Revenue generated']
    colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0]*len(labels),colors=colors[:len(sizes)])
    fig.patch.set_edgecolor('black')  
    fig.patch.set_linewidth(1) 
    plt.title('Revenue Distribution by Location',weight='bold',fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

with col2:
    #Chart6
    supplier_revenue=filtered_df.groupby('Supplier name')['Revenue generated'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10,7), facecolor='none')
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
    

col1,col2=st.columns(2)
with col1:
    #Chart7
    leadtime=filtered_df.groupby('Supplier name')['Lead times'].mean().reset_index()
    fig,ax=plt.subplots(figsize=(12,8))
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
    fig,ax=plt.subplots(figsize=(12,8))
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

col1,col2=st.columns(2)
with col1:
    #Chart9
    unit_sold_stat=filtered_df.groupby('Product type')['Number of products sold'].sum().reset_index()
    fig,ax=plt.subplots(figsize=(12,8))
    ax=sns.barplot(data=unit_sold_stat,x='Product type',y='Number of products sold',palette='Reds')
    bars=ax.containers[0]
    ax.bar_label(bars,fmt='%.f')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.xlabel('Product type')
    plt.ylabel('Number of products sold')
    plt.title('Units Sold by Product Type',weight='bold',fontsize=12)
    st.pyplot(fig)

with col2:
    #Chart10
    transportation_stats=filtered_df.groupby('Transportation modes')['Shipping costs'].mean().reset_index()
    fig,ax=plt.subplots(figsize=(12,8))
    ax=sns.barplot(data=transportation_stats,y='Transportation modes',x='Shipping costs',palette='viridis')
    bars=ax.containers[0] 
    ax.bar_label(bars,fmt='%.f')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.xlabel('Shipping costs')
    plt.ylabel('Transportation modes')
    plt.title('Shipping Cost by Transportation Mode',weight='bold',fontsize=12)
    st.pyplot(fig)

col1,col2=st.columns(2)
with col1:
    #Chart11
    location_revenue=filtered_df.groupby('Location')['Revenue generated'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(5,3), facecolor='none')
    labels=location_revenue['Location']
    sizes=location_revenue['Revenue generated']
    colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    plt.pie(sizes,labels=labels,autopct='%1.1f%%',explode=[0]*len(labels),colors=colors[:len(sizes)])
    fig.patch.set_edgecolor('black')  
    fig.patch.set_linewidth(1) 
    plt.title('Revenue Distribution by Location',weight='bold',fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

with col2:
    #Chart12
    carrier_statistics=filtered_df.groupby('Shipping carriers')[['Shipping times','Shipping costs']].mean().reset_index()
    fig,ax=plt.subplots(figsize=(12,10))
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
    

col1,col2=st.columns(2)
with col1:
    #Chart13
    transportation_stats=filtered_df.groupby('Transportation modes')['Order quantities'].sum().reset_index()
    plt.figure(figsize=(12,8),facecolor='none')
    sns.heatmap(transportation_stats['Order quantities'].values.reshape(-1,1),annot=True,fmt='.0f',cmap='YlGnBu',
            yticklabels=transportation_stats['Transportation modes'],xticklabels=['Order quantities'])
    plt.xlabel('')
    plt.ylabel('Transportation modes')
    plt.title('Total Order Quantity by Transportation Mode',weight='bold',fontsize=12)
    st.pyplot(plt)

with col2:
    #Chart14
    fig,ax=plt.subplots(figsize=(12,8))
    sns.lineplot(data=filtered_df,x='Price',y='Revenue generated',marker='o',color='blue')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    plt.title('Revenue Generated by Price Range',weight='bold',fontsize=12)
    plt.xlabel('Price')
    plt.ylabel('Revenue Generated')
    st.pyplot(fig)

col1,col2=st.columns(2)
with col1:
    #Chart15
    locationstats=filtered_df.groupby('Location')['Order quantities'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12,8))
    ax=sns.barplot(data=locationstats,x='Location',y='Order quantities',palette='muted')
    bars=ax.containers[0]
    ax.bar_label(bars,fmt='%.f')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.xlabel('Location')
    plt.ylabel('Order Quantities')
    plt.title('Order Quantities by Location',weight='bold',fontsize=12)
    st.pyplot(fig)

with col2:
    #Chart16
    filtered_df['Profit']=filtered_df['Revenue generated']-filtered_df['Manufacturing costs']
    profitby_prod=filtered_df.groupby('Product type')['Profit'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12,8))
    ax=sns.barplot(data=profitby_prod,x='Product type',y='Profit',palette='Paired')
    bars=ax.containers[0]
    ax.bar_label(bars,fmt='%.f')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.xlabel('Product type')
    plt.ylabel('Profit')
    plt.title('Overall Profitability by Product Type',weight='bold',fontsize=12)
    st.pyplot(fig)

st.markdown("<p style='text-align: center; font-size: 12px;'>Made by Nikita Mendhe | <a href='www.linkedin.com/in/nikita-mendhe-2067b5210' target='_blank'>LinkedIn</a></p>", unsafe_allow_html=True)
