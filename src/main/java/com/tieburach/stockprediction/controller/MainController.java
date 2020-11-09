package com.tieburach.stockprediction.controller;

import com.tieburach.stockprediction.model.WIGDataEntity;
import com.tieburach.stockprediction.prediction.StockDataSetIterator;
import com.tieburach.stockprediction.prediction.StockPricePrediction;
import com.tieburach.stockprediction.repository.WIGDataRepository;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

import java.time.LocalDate;
import java.util.List;

@Controller
@RequestMapping
public class MainController {
    private static final LocalDate START = LocalDate.of(2000, 1, 1);
    private static final LocalDate END = LocalDate.of(2018, 1, 1);
    private final WIGDataRepository repository;
    private final StockPricePrediction pricePrediction;
    private MultiLayerNetwork net = null;
    private StockDataSetIterator iterator = null;

    @Autowired
    public MainController(WIGDataRepository repository, StockPricePrediction pricePrediction) {
        this.repository = repository;
        this.pricePrediction = pricePrediction;
    }
    @GetMapping("/initialize")
    public ResponseEntity<?> initialize() {
        List<WIGDataEntity> entities = repository.getAllByDateBetween(START, END);
        iterator = pricePrediction.initializeIterator(entities);
        net = pricePrediction.initialize(iterator);
        return ResponseEntity.ok("Net was initialized and trained.");
    }

    @GetMapping("/predict/{daysAhead}")
    public ResponseEntity<?> existingMethod(@PathVariable("daysAhead") int daysAhead) {
        if (net == null || iterator == null) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body("Net wasn't initialized.");
        }
        pricePrediction.predict(net, iterator, daysAhead);
        return ResponseEntity.ok().build();
    }
}
