package com.tieburach.stockprediction.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Getter
@Setter
@Component
public class NeuralNetProperties {
    @Value("${net.bptt}")
    private int bptt;
    @Value("${net.batch_size}")
    private int batchSize;
    @Value("${net.days_to_show}")
    private int daysToShow;
    @Value("1")
    private int epochs;
    @Value("${net.days_ahead}")
    private int daysAhead;
    @Value("${net.iterations}")
    private int iterations;
}
