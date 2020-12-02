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


pass <- getPass()
con <- dbConnect(RPostgres::Postgres(),
                 dbname = "postgres",
                 host = "localhost",
                 port = "5432",
                 user = "postgres", password = pass)


data <- dbGetQuery(con, "SELECT * from stockvalues where date between '2010-01-01' and '2020-01-01'")


wig_close <- data[, c(5)]

model <- auto.arima(wig_close)
par(mfrow=c(1,1))
acf(model$residuals)
boxresult <- LjungBoxTest(model$residuals,k=2,StartLag=1)
boxresult
plot(boxresult[,3],main= "Ljung-Box Q Test", ylab= "P-values", xlab= "Lag")
qqnorm(model$residuals)
qqline(model$residuals)


pacf(diff(wig_close))




adf.test(wig_close, alternative = "stationary")
adf.test(diff(wig_close), alternative = "stationary")


par(mfrow=c(1,2))
acf(wig_close)
pacf(wig_close)
acf(diff(wig_close))
pacf(diff(wig_close))


kpss.test(wig_close, null="Trend")
kpss.test(diff(wig_close), null = "Trend")




tsdata <- ts(wig_close, frequency = 254, start=c(2010,1))
components.ts = decompose(tsdata)
plot(components.ts)

plot (tsdata)

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
  arimaModel_1 = auto.arima(train_series)
  forecast1 = forecast(arimaModel_1, daysAhead);
  predicted = forecast1$mean[daysAhead]
  actualVector <- append (actualVector, actual)
  predictsVector <- append (predictsVector, predicted)
  mape = mape + abs((actual - predicted)/actual)
  i = i + 1
}

mape = mape / length(test_series)
print(mape * 100)


kroki <- 1:length(actualVector)

plot(kroki, ylab = "Wartosc indeksu zamkniecia WIG20", xlab= "Indeks", actualVector, col="blue", type='l')
lines(kroki, predictsVector, col="red",pch="+", type ='l')
legend(kroki[length(daty) -115], 2430,legend=c("Aktualne wartosci","Wartosci prognozowane"), col=c("blue","red"), lty=1, ncol=1)





