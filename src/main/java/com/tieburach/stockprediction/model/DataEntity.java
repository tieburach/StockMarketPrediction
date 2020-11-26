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
@Table(name = "\"stockvalues\"")
public class DataEntity {
    @Id
    private LocalDate date;
    private Double open;
    private Double close;
    private Double high;
    private Double low;
    private Double EUR_Otwarcie;
    private Double EUR_Najwyzszy;
    private Double EUR_Najnizszy;
    private Double EUR_Zamkniecie;
    private Double PKO_Otwarcie;
    private Double PKO_Najwyzszy;
    private Double PKO_Najnizszy;
    private Double PKO_Zamkniecie;
    private Double USD_Otwarcie;
    private Double USD_Najwyzszy;
    private Double USD_Najnizszy;
    private Double USD_Zamkniecie;
    private Double KGH_Otwarcie;
    private Double KGH_Najwyzszy;
    private Double KGH_Najnizszy;
    private Double KGH_Zamkniecie;
}
