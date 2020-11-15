package com.tieburach.stockprediction.controller;

import com.tieburach.stockprediction.model.DataEntity;
import com.tieburach.stockprediction.model.WIGDataEntity;
import com.tieburach.stockprediction.prediction.StockPricePrediction;
import com.tieburach.stockprediction.repository.WIGDataRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import java.time.LocalDate;
import java.util.List;

@Controller

@RequestMapping
public class MainController {
    private static final LocalDate START = LocalDate.of(2002, 1, 1);
    private static final LocalDate END = LocalDate.of(2013, 1, 1);
    private final WIGDataRepository repository;
    private final StockPricePrediction pricePrediction;

    @Autowired
    public MainController(WIGDataRepository repository, StockPricePrediction pricePrediction) {
        this.repository = repository;
        this.pricePrediction = pricePrediction;
    }

    @GetMapping("/predict")
    public ResponseEntity<?> predict() {
        List<WIGDataEntity> entities = repository.getAllByDateBetween(START, END);
        List<DataEntity> features = repository.getAllFeatures();
        pricePrediction.initialize(entities);
        pricePrediction.predict();
        return ResponseEntity.ok().build();
    }
}
