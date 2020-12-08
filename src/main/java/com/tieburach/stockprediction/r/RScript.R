library(ggplot2)
library(tseries)
library(forecast)
library(RODBC)
library(rmarkdown)
require(RPostgreSQL)
require(RPostgres)
require(DBI)
library(rpart)
library(getPass)
library(FitAR)


pass <- getPass()
con <- dbConnect(RPostgres::Postgres(),
                 dbname = "postgres",
                 host = "localhost",
                 port = "5432",
                 user = "postgres", password = pass)


#load the values
data <- dbGetQuery(con, "SELECT * from stockvalues where date between '2010-01-01' and '2020-01-01'")
wig_close <- data[, c(5)]

#plot the data
plot(wig_close)

#check with ADF (augmented Dickey-Fuller) test
adf.test(wig_close, alternative = "stationary")

#check with ADF test for differenced series
adf.test(diff(wig_close), alternative = "stationary")

#check with KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test
kpss.test(wig_close, null = "Trend")

#check with KPSS test for differenced series
kpss.test(diff(wig_close), null = "Trend")

#plot ACF (auto-correlation function) for differenced series
acf(diff(wig_close))

#plot PACF (partial auto-correlation function) for differenced series
pacf(diff(wig_close))

#look at proposed arima model from auto.arima
model <- auto.arima(wig_close)
summary(model)


#print corelogram for residuals of model
acf(model$residuals)
boxresult <- LjungBoxTest(model$residuals,k=2,StartLag=1)
boxresult
plot(boxresult[,3],main= "Ljung-Box Q Test", ylab= "P-values", xlab= "Lag")

#look at q-q chart
plot(qqnorm(model$residuals))
line(qqline(model$residuals))

#look at seasonal/trend/random decomposition
tsdata <- ts(wig_close, frequency = 254, start=c(2010,1))
components.ts = decompose(tsdata)
plot(components.ts)


#Make predictions
daysAhead = 1
test_length = floor(0.8 * length(wig_close))
test_series = wig_close[(test_length + daysAhead):length(wig_close)]
i = 0
mape = 0
actualVector = c()
predictsVector = c()
for (actual in test_series) {
  train_series = wig_close[1:(test_length + i)]
  arimaModel_1 = arima(train_series, order = c(3, 1, 4), include.mean = FALSE)
  forecast1 = forecast(arimaModel_1, daysAhead);
  predicted = forecast1$mean[daysAhead]
  actualVector <- append (actualVector, actual)
  predictsVector <- append (predictsVector, predicted)
  mape = mape + abs((actual - predicted)/actual)
  i = i + 1
}
mape = mape / length(test_series)
print()
print(mape * 100)



steps <- 1:length(actualVector)
plot(steps, ylab = "Wartosc indeksu zamkniecia WIG20", xlab= "Indeks", actualVector, col="blue", type='l')
lines(steps, predictsVector, col="red",pch="+", type ='l')
legend(steps[length(steps) -115], 2430,legend=c("Aktualne wartosci","Wartosci prognozowane"), col=c("blue","red"), lty=1, ncol=1)