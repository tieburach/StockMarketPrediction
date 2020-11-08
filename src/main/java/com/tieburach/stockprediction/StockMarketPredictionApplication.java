package com.tieburach.stockprediction;

import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;

@SpringBootApplication
public class StockMarketPredictionApplication {
    public static void main(String[] args) {
        SpringApplicationBuilder builder = new SpringApplicationBuilder(StockMarketPredictionApplication.class);
        builder.headless(false);
        builder.run(args);
    }
}
