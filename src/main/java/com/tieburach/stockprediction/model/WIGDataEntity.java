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
@Table(name = "\"WIG\"")
public class WIGDataEntity {
    @Id
    private LocalDate date;
    private Double open;
    private Double close;
    private Double high;
    private Double low;
    private Double wol;

}
