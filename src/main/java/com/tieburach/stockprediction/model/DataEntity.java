package com.tieburach.stockprediction.model;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "\"FEATURES\"")
public class DataEntity {
    @Id
    private LocalDate date;
    private Double open;
    private Double close;
    private Double high;
    private Double low;
    private Double spx_open;
    private Double spx_close;
    private Double spx_high;
    private Double spx_low;
    private Double dax_open;
    private Double dax_close;
    private Double dax_high;
    private Double dax_low;
}
