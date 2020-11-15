package com.tieburach.stockprediction.prediction;

import com.tieburach.stockprediction.model.DataEntity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;
import java.util.List;

public class StockDataSetIterator implements DataSetIterator {
    public static final int VECTOR_SIZE = 4;
    private final int batchSize;
    private final int bptt;
    private final int dayAhead;
    private final double[] minValuesInFeature;
    private final double[] maxValuesInFeature;
    private final LinkedList<Integer> startOffset = new LinkedList<>();
    private final List<DataEntity> train;

    public StockDataSetIterator(List<DataEntity> dataEntities, int batchSize, int bptt, int dayAhead,
                                double[] minValuesInFeature, double[] maxValuesInFeature) {
        this.dayAhead = dayAhead;
        this.batchSize = batchSize;
        this.bptt = bptt;
        this.train = dataEntities;
        this.maxValuesInFeature = maxValuesInFeature;
        this.minValuesInFeature = minValuesInFeature;
        initializeOffsets();
    }

    private void initializeOffsets() {
        startOffset.clear();
        int window = bptt + dayAhead;
        for (int i = 0; i < train.size() - window; i++) {
            startOffset.add(i);
        }
    }

    @Override
    public DataSet next(int num) {
        int actualMiniBatchSize = Math.min(num, startOffset.size());
        INDArray input = Nd4j.create(new int[]{actualMiniBatchSize, VECTOR_SIZE, bptt}, 'f');
        INDArray label = Nd4j.create(new int[]{actualMiniBatchSize, 1, bptt}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int start = startOffset.removeFirst();
            DataEntity currentRecord = train.get(start);
            for (int i = start; i < start + bptt; i++) {
                int column = i - start;
                populateINDArray(input, index, currentRecord, column);
                populateINDArrayLabel(label, index, train.get(i + dayAhead).getClose(), column);
                currentRecord = train.get(i + 1);
            }
            if (startOffset.size() == 0) {
                break;
            }
        }
        return new DataSet(input, label);
    }

    private void populateINDArray(INDArray array, int index, DataEntity dataElement, int column) {
        array.putScalar(new int[]{index, 0, column}, (dataElement.getOpen() - minValuesInFeature[0]) / (maxValuesInFeature[0] - minValuesInFeature[0]));
        array.putScalar(new int[]{index, 1, column}, (dataElement.getClose() - minValuesInFeature[1]) / (maxValuesInFeature[1] - minValuesInFeature[1]));
        array.putScalar(new int[]{index, 2, column}, (dataElement.getLow() - minValuesInFeature[2]) / (maxValuesInFeature[2] - minValuesInFeature[2]));
        array.putScalar(new int[]{index, 3, column}, (dataElement.getHigh() - minValuesInFeature[3]) / (maxValuesInFeature[3] - minValuesInFeature[3]));
//        array.putScalar(new int[]{index, 4, column}, (dataElement.getDax_open() - minValuesInFeature[4]) / (maxValuesInFeature[4] - minValuesInFeature[4]));
//        array.putScalar(new int[]{index, 5, column}, (dataElement.getDax_close() - minValuesInFeature[5]) / (maxValuesInFeature[5] - minValuesInFeature[5]));
//        array.putScalar(new int[]{index, 6, column}, (dataElement.getDax_low() - minValuesInFeature[6]) / (maxValuesInFeature[6] - minValuesInFeature[6]));
//        array.putScalar(new int[]{index, 7, column}, (dataElement.getDax_high() - minValuesInFeature[7]) / (maxValuesInFeature[7] - minValuesInFeature[7]));
//        array.putScalar(new int[]{index, 8, column}, (dataElement.getSpx_open() - minValuesInFeature[8]) / (maxValuesInFeature[8] - minValuesInFeature[8]));
//        array.putScalar(new int[]{index, 9, column}, (dataElement.getSpx_close() - minValuesInFeature[9]) / (maxValuesInFeature[9] - minValuesInFeature[9]));
//        array.putScalar(new int[]{index, 10, column}, (dataElement.getSpx_low() - minValuesInFeature[10]) / (maxValuesInFeature[10] - minValuesInFeature[10]));
//        array.putScalar(new int[]{index, 11, column}, (dataElement.getSpx_high() - minValuesInFeature[11]) / (maxValuesInFeature[11] - minValuesInFeature[11]));
    }

    private void populateINDArrayLabel(INDArray array, int index, Double value, int column) {
        array.putScalar(new int[]{index, 0, column}, (value - minValuesInFeature[1]) / (maxValuesInFeature[1] - minValuesInFeature[1]));
    }

    @Override
    public int totalExamples() {
        return train.size() - bptt - dayAhead;
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


}
