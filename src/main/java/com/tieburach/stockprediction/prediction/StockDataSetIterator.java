package com.tieburach.stockprediction.prediction;

import com.tieburach.stockprediction.model.WIGDataEntity;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class StockDataSetIterator implements DataSetIterator {
    public static final int VECTOR_SIZE = 5;
    private final int batchSize;
    private final int bptt;
    private final double[] minValuesInFeature = new double[VECTOR_SIZE];
    private final double[] maxValuesInFeature = new double[VECTOR_SIZE];
    private final LinkedList<Integer> startOffset = new LinkedList<>();
    private final List<WIGDataEntity> train;
    private final List<Pair<INDArray, INDArray>> test;
    private final List<String> datesList = new ArrayList<>();

    public StockDataSetIterator(List<WIGDataEntity> wigDataEntities, int batchSize, int bptt, int daysToShow) {
        this.batchSize = batchSize;
        this.bptt = bptt;

        for (int i = 0; i < maxValuesInFeature.length; i++) {
            maxValuesInFeature[i] = Double.MIN_VALUE;
            minValuesInFeature[i] = Double.MAX_VALUE;
        }

        for (WIGDataEntity entity : wigDataEntities) {
            setMinAndMax(entity.getOpen(), 0);
            setMinAndMax(entity.getClose(), 1);
            setMinAndMax(entity.getLow(), 2);
            setMinAndMax(entity.getHigh(), 3);
            setMinAndMax(entity.getWol(), 4);
        }

        int splitNumber = wigDataEntities.size() - (bptt + 1 + daysToShow);
        prepareDates(wigDataEntities.subList(wigDataEntities.size() - daysToShow - 1, wigDataEntities.size()));
        train = wigDataEntities.subList(0, splitNumber);
        test = generateTestDataSet(wigDataEntities.subList(splitNumber, wigDataEntities.size()));

        initializeOffsets();
    }

    public List<String> getDatesList() {
        return datesList;
    }

    private void prepareDates(List<WIGDataEntity> subList) {
        for (WIGDataEntity entity : subList) {
            datesList.add(entity.getDate().toString());
        }
    }

    private void setMinAndMax(Double value, int i) {
        if (value > maxValuesInFeature[i]) {
            maxValuesInFeature[i] = value;
        }
        if (value < minValuesInFeature[i]) {
            minValuesInFeature[i] = value;
        }
    }

    private void initializeOffsets() {
        startOffset.clear();
        int window = bptt + 1;
        for (int i = 0; i < train.size() - window; i++) {
            startOffset.add(i);
        }
    }

    public List<Pair<INDArray, INDArray>> getTestDataSet() {
        return test;
    }

    public double[] getMaxValuesInFeature() {
        return maxValuesInFeature;
    }

    public double[] getMinValuesInFeature() {
        return minValuesInFeature;
    }

    @Override
    public DataSet next(int num) {
        int actualMiniBatchSize = Math.min(num, startOffset.size());
        INDArray input = Nd4j.create(new int[]{actualMiniBatchSize, VECTOR_SIZE, bptt}, 'f');
        INDArray label = Nd4j.create(new int[]{actualMiniBatchSize, VECTOR_SIZE, bptt}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int start = startOffset.removeFirst();
            WIGDataEntity currentRecord = train.get(start);
            WIGDataEntity nextRecord;
            for (int i = start; i < start + bptt; i++) {
                int column = i - start;
                populateINDArray(input, index, currentRecord, column);
                nextRecord = train.get(i + 1);
                populateINDArray(label, index, nextRecord, column);
                currentRecord = nextRecord;
            }
            if (startOffset.size() == 0) {
                break;
            }
        }
        return new DataSet(input, label);
    }

    private void populateINDArray(INDArray array, int index, WIGDataEntity dataElement, int column) {
        array.putScalar(new int[]{index, 0, column}, (dataElement.getOpen() - minValuesInFeature[0]) / (maxValuesInFeature[0] - minValuesInFeature[0]));
        array.putScalar(new int[]{index, 1, column}, (dataElement.getClose() - minValuesInFeature[1]) / (maxValuesInFeature[1] - minValuesInFeature[1]));
        array.putScalar(new int[]{index, 2, column}, (dataElement.getLow() - minValuesInFeature[2]) / (maxValuesInFeature[2] - minValuesInFeature[2]));
        array.putScalar(new int[]{index, 3, column}, (dataElement.getHigh() - minValuesInFeature[3]) / (maxValuesInFeature[3] - minValuesInFeature[3]));
        array.putScalar(new int[]{index, 4, column}, (dataElement.getWol() - minValuesInFeature[4]) / (maxValuesInFeature[4] - minValuesInFeature[4]));
    }

    @Override
    public int totalExamples() {
        return train.size() - bptt - 1;
    }

    @Override
    public int inputColumns() {
        return VECTOR_SIZE;
    }

    @Override
    public int totalOutcomes() {
        return VECTOR_SIZE;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        initializeOffsets();
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return totalExamples() - startOffset.size();
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public boolean hasNext() {
        return startOffset.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    private List<Pair<INDArray, INDArray>> generateTestDataSet(List<WIGDataEntity> stockDataList) {
        int window = bptt + 1;
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();
        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{bptt, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + bptt; j++) {
                WIGDataEntity stock = stockDataList.get(j);
                input.putScalar(new int[]{j - i, 0}, (stock.getOpen() - minValuesInFeature[0]) / (maxValuesInFeature[0] - minValuesInFeature[0]));
                input.putScalar(new int[]{j - i, 1}, (stock.getClose() - minValuesInFeature[1]) / (maxValuesInFeature[1] - minValuesInFeature[1]));
                input.putScalar(new int[]{j - i, 2}, (stock.getLow() - minValuesInFeature[2]) / (maxValuesInFeature[2] - minValuesInFeature[2]));
                input.putScalar(new int[]{j - i, 3}, (stock.getHigh() - minValuesInFeature[3]) / (maxValuesInFeature[3] - minValuesInFeature[3]));
                input.putScalar(new int[]{j - i, 4}, (stock.getWol() - minValuesInFeature[4]) / (maxValuesInFeature[4] - minValuesInFeature[4]));
            }
            WIGDataEntity stock = stockDataList.get(i + bptt);
            INDArray label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f');
            label.putScalar(new int[]{0}, stock.getOpen());
            label.putScalar(new int[]{1}, stock.getClose());
            label.putScalar(new int[]{2}, stock.getLow());
            label.putScalar(new int[]{3}, stock.getHigh());
            label.putScalar(new int[]{4}, stock.getWol());
            test.add(new Pair<>(input, label));
        }
        return test;
    }

}
