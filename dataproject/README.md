# Data analysis project

Our project is titled **Alternative Philips curve** explores the use of the consumer price index for beer as a substitute for inflation in depicting the traditional Philips curve. The Philips curve posits that higher inflation leads to lower unemployment rates and vice versa. However, our findings indicate that the consumer price index for beer is a less effective alternative for depicting the traditional Philips curve, as the numeric correlation between the consumer price index for beer and the gross unemployment rate is lower compared to when inflation is used.

The **results** of the project can be seen from running [beer.ipynb](beer.ipynb).

We apply the **following datasets** by using DstApi:

DstApi("PRIS111") and DstApi("AUS07")

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

`pip install pandas-datareader`
`pip install git+https://github.com/alemartinello/dstapi`