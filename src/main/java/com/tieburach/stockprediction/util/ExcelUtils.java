package com.tieburach.stockprediction.util;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.FileOutputStream;
import java.time.LocalDateTime;
import java.util.List;

public class ExcelUtils {
    public static void writeToExcel(INDArray[] predicts, INDArray[] actuals, List<String> datesList) {
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Prediction");
        Row row = sheet.createRow(0);
        Cell cell = row.createCell(0);
        cell.setCellValue("DATE");
        cell = row.createCell(1);
        cell.setCellValue("ACTUAL CLOSE");
        cell = row.createCell(2);
        cell.setCellValue("PREDICTED CLOSE");
        int rowCount = 1;
        for (int i = 0; i < predicts.length; i++) {
            row = sheet.createRow(rowCount);
            cell = row.createCell(0);
            cell.setCellValue(datesList.get(i));
            cell = row.createCell(1);
            cell.setCellValue(actuals[i].getDouble(1));
            cell = row.createCell(2);
            cell.setCellValue(predicts[i].getDouble(1));
            rowCount++;
        }
        for (int i = 0; i < 3; i++) {
            sheet.autoSizeColumn(i);
        }

        try (FileOutputStream outputStream = new FileOutputStream(
                "PREDICTION_" + LocalDateTime.now().getHour() + "_" + LocalDateTime.now().getMinute() + ".xlsx")) {
            workbook.write(outputStream);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
