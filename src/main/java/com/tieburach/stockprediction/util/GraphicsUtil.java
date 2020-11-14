package com.tieburach.stockprediction.util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class GraphicsUtil {
    public static void draw(double[] predictedValues, double[] actualValues) {
        int[] index = new int[predictedValues.length];
        for (int i = 0; i < predictedValues.length; i++) {
            index[i] = i;
        }
        int min = minValue(predictedValues, actualValues);
        int max = maxValue(predictedValues, actualValues);
        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, index, predictedValues, "Predicts");
        addSeries(dataSet, index, actualValues, "Actuals");
        ChartFactory.setChartTheme(StandardChartTheme.createLegacyTheme());
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Prediction",
                "Index",
                "WIG20 - CLOSE",
                dataSet,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        XYPlot xyPlot = chart.getXYPlot();
        final NumberAxis domainAxis = (NumberAxis) xyPlot.getDomainAxis();
        domainAxis.setRange(index[0], index[index.length - 1] + 2);
        domainAxis.setTickUnit(new NumberTickUnit(Math.floor((double) index.length / 10)));
        domainAxis.setVerticalTickLabels(true);
        final NumberAxis rangeAxis = (NumberAxis) xyPlot.getRangeAxis();
        rangeAxis.setRange(min, max);
        rangeAxis.setTickUnit(new NumberTickUnit(Math.floor(((double) max - (double) min) / 10)));
        final ChartPanel panel = new ChartPanel(chart);
        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        f.pack();
        f.setVisible(true);
    }

    private static void addSeries(final XYSeriesCollection dataSet, int[] x, double[] y, final String label) {
        final XYSeries s = new XYSeries(label);
        for (int j = 0; j < x.length; j++) {
            s.add(x[j], y[j]);
        }
        dataSet.addSeries(s);
    }

    private static int minValue(double[] predicts, double[] actuals) {
        double min = Integer.MAX_VALUE;
        for (int i = 0; i < predicts.length; i++) {
            if (min > predicts[i]) {
                min = predicts[i];
            }
            if (min > actuals[i]) {
                min = actuals[i];
            }
        }
        return (int) (min * 0.95);
    }

    private static int maxValue(double[] predicts, double[] actuals) {
        double max = Integer.MIN_VALUE;
        for (int i = 0; i < predicts.length; i++) {
            if (max < predicts[i]) {
                max = predicts[i];
            }
            if (max < actuals[i]) {
                max = actuals[i];
            }
        }
        return (int) (max * 1.05);
    }

}