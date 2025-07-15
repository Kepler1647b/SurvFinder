###size 900 500
library(survival)
library(showtext) 
library(stringr)
library(ggplot2)
font_add("Arial_a", '/Library/Fonts/Arial Unicode.ttf')
showtext_auto()
csv_inn_sur = read.table('/data_all_inner.csv', sep = ',', header = TRUE)
csv_inn_sur$censor = as.integer(csv_inn_sur$censor)
csv_inn_sur$time = csv_inn_sur$time
coln = colnames(csv_inn_sur)
coln[coln == 'pvalue'] = 'MVNet'
coln[coln == 'differentiation'] = 'Differentiation'
coln[coln == 'hole'] = 'Perforation'
coln[coln == 'perineural'] = 'PNI'
coln[coln == 'vascular'] = 'VI'
coln[coln == 'tumor_budding'] = 'Tumor_budding'
coln[coln == 'signet'] = 'SRCC'
coln[coln == 'mucinous'] = 'MAC'
coln[coln == 'classification'] = 'Classification'
coln[coln == 'location'] = 'Primary_site'
coln[coln == 'lympho_node_number'] = 'LNS'
coln[coln == 'grade'] = 'Grading'
coln[coln == 'stage_T'] = 'T_stage'
coln[coln == 'mmr2'] = 'MMR'
coln[coln == 'CEA'] = 'CEA'
coln[coln == 'CA199'] = 'CA199'
coln[coln == 'sex'] = 'Gender'
coln[coln == 'age'] = 'Age'
coln[coln == 'adj_after'] = 'ACT'
colnames(csv_inn_sur) = coln
csv_inn_sur$Age = as.integer(csv_inn_sur$Age >= 65)
csv_inn_sur$CA199 = as.integer(csv_inn_sur$CA199 > 37)
csv_inn_sur$CEA = as.integer(csv_inn_sur$CEA > 5)
csv_inn_sur$LNS = as.integer(csv_inn_sur$LNS >= 12)
csv_inn_sur$Differentiation = as.integer(csv_inn_sur$Differentiation)
csv_inn_sur$BD3 = as.integer(csv_inn_sur$Tumor_budding)
csv_inn_sur$BD3[csv_inn_sur$BD3 == 0] = 1
csv_inn_sur$Grading = as.integer(csv_inn_sur$Grading)
covariates <- c("MMR", 'BD3', 'Tumor_budding', "CEA", "CA199", 'T_stage', 'LNS', 'Primary_site', 'Classification', 'MAC', 'SRCC', 'VI', 'PNI', 'Perforation','Grading', 'Differentiation', 'MVNet')

library(wesanderson)
pal <- wes_palette("Zissou1", 18, type = "continuous")
library(forestplot)
td = csv_inn_sur
pFilter=1
pfil = 0.1
outResult=data.frame() 
sigGenes=c("censor","time")
sigG = c()
foc_feat <- c("MMR", 'BD3', 'Tumor_budding',"CEA", "CA199", 'T_stage', 'LNS', 'Primary_site', 'Classification', 'MAC', 'SRCC', 'VI', 'PNI', 'Perforation', 'Grading','Differentiation', 'MVNet')
for(i in foc_feat){ 
  tdcox <- coxph(Surv(time, censor) ~ td[,i], data = td)
  tdcoxSummary = summary(tdcox) 
  pvalue=tdcoxSummary$coefficients[,"Pr(>|z|)"] 
  pvalue
  if(pvalue<pfil){
    sigG = c(sigG, i)
  }
  if(pvalue<pFilter){
    sigGenes=c(sigGenes,i)
    outResult=rbind(outResult,
                    cbind(id=i,
                          HR=tdcoxSummary$conf.int[,"exp(coef)"],
                          L95CI=tdcoxSummary$conf.int[,"lower .95"],
                          H95CI=tdcoxSummary$conf.int[,"upper .95"],
                          pvalue=tdcoxSummary$coefficients[,"Pr(>|z|)"],
                          interv = paste0(round(tdcoxSummary$conf.int[,'exp(coef)'], 2), ' (', round(tdcoxSummary$conf.int[,'lower .95'], 2), ', ', round(tdcoxSummary$conf.int[,'upper .95'], 2), ')'))
    )
  }
}
UniCoxSurSigGeneExp=td[,sigGenes] 
UniCoxSurSigGeneExp=cbind(id=row.names(UniCoxSurSigGeneExp),UniCoxSurSigGeneExp)
outResult$id_sp = outResult$id
outResult$id_sp[str_detect(outResult$id, '\\_')] = str_replace(outResult$id[str_detect(outResult$id, '\\_')], '\\_', ' ')
outResult$id_sp = factor(outResult$id_sp, levels = as.character(outResult$id_sp))

outResult$pvalue = round(as.numeric(outResult$pvalue), 3)
outResult$pvalue[outResult$pvalue < 0.001] = '< 0.001'
outResult$pvalue[outResult$pvalue >0.99] = NA
outResult$L95CI[outResult$L95CI ==0] = NA
outResult$H95CI[outResult$H95CI >9999] = NA
outResult$pvalue = as.character(outResult$pvalue)
p <- ggplot(outResult, aes(as.numeric(outResult$HR), y = id_sp, col = id_sp))

p_cox = p +  #scale_color_brewer(palettes = pal)+
  geom_vline(aes(xintercept = 1), linetype = 'dotted') +
  
  geom_errorbarh(aes(xmax =as.numeric(outResult$H95CI), xmin = as.numeric(outResult$L95CI)), height = 0.3, size = 0.5, color = 'black') + geom_point(size=4.5, color = 'white') + geom_point(size=3.2) +
  
  
  
  coord_cartesian(xlim = c(0.07, 20), # This focuses the x-axis on the range of interest
                  ylim = c(1, 18),
                  clip = 'off') +   # This keeps the labels from disappearing
  xlab('HR') + ylab('Univariate Cox regression')+theme_bw()+theme(panel.grid=element_blank())+
  theme(legend.position = 'None',plot.margin = unit(c(1,8,1,1), "lines"),
        legend.title = element_text( family = 'Arial_a', size = 7), legend.text = element_text( family = 'Arial_a', size = 7), axis.text.y = element_text(hjust = 1, family = 'Arial_a', size = 7, colour = 'black'), axis.text.x = element_text(family = 'Arial_a', size = 7, colour = 'black'), axis.title.x = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = -1), axis.title.y = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = 1))+
  geom_text(aes(x = 600, label=outResult$pvalue), color = 'black',hjust=1, size = 7/.pt, family = 'Arial_a')+
  geom_text(aes(x =40, label=outResult$interv), color = 'black',hjust=0, size = 7/.pt, family = 'Arial_a')+
  annotate("text", x = 43 , y = -0.5,label = "HR (95% CI)", color = 'black',hjust=0, size = 7/.pt, family = 'Arial_a')+
  annotate("text", x =610 , y = -0.5,label = "p value", color = 'black',hjust=1, size = 7/.pt, family = 'Arial_a')+
  scale_x_continuous(limits= c(0.02, 10000), breaks= c(0.1, 0.2, 0.5, 2, 5, 10), trans = 'log10') +scale_color_my(palette = 'mixed')
p_cox
id_d = as.data.frame(outResult$id)
colnames(id_d)= c('id')

library(tableone)  
foo <- as.formula(paste0("Surv(time, censor) ~ ", paste(sigG, collapse = ' + ')))
mul_cox<-coxph(foo,data=td)
mul_cox1 <- summary(mul_cox)
colnames(mul_cox1$conf.int)
multi1<-as.data.frame(round(mul_cox1$conf.int[, c(1, 3, 4)], 2))
multi2<-ShowRegTable(mul_cox, 
                     exp=TRUE, 
                     digits=2, 
                     pDigits =3,
                     printToggle = TRUE, 
                     quote=FALSE, 
                     ciFun=confint)
result <-cbind(multi1,multi2);result
result$p[result$p == '<0.001'] = '< 0.001'
result<-tibble::rownames_to_column(result, var = "Characteristics");result
id_d$ind = 1:nrow(id_d)
result_na = merge(id_d, result, by.x = 'id', by.y = 'Characteristics', all.x = TRUE)
result_na = result_na[order(result_na$ind),]
result_na$`exp(coef) [confint]`[is.na(result_na$`exp(coef) [confint]`)] = 'Not included'
t_a = result_na$`exp(coef) [confint]`
result_na$`exp(coef) [confint]`[str_detect(t_a, '\\[')] = str_replace(t_a[str_detect(t_a, '\\[')], '\\[', '(')
t_a = result_na$`exp(coef) [confint]`
result_na$`exp(coef) [confint]`[str_detect(t_a, '\\]')] = str_replace(t_a[str_detect(t_a, '\\]')], '\\]', ')')

result_na$id_sp = result_na$id
result_na$id_sp[str_detect(result_na$id, '\\_')] = str_replace(result_na$id[str_detect(result_na$id, '\\_')], '\\_', ' ')
result_na$id_sp = factor(result_na$id_sp, levels = as.character(result_na$id_sp))

pm <- ggplot(result_na, aes(as.numeric(result_na$`exp(coef)`), y = id_sp, col = id_sp))

pm_mul = pm +  #scale_color_brewer(palettes = pal)+
  geom_vline(aes(xintercept = 1), linetype='dotted') +
  
  geom_errorbarh(aes(xmax =as.numeric(result_na$`lower .95`), xmin = as.numeric(result_na$`upper .95`)), height = 0.3, size = 0.5, color = 'black') + geom_point(size=4.5, color = 'white') + geom_point(size=3.2) +
  
  
  
  
  
  coord_cartesian(xlim = c(0.07, 20), # This focuses the x-axis on the range of interest#0.03,28
                  ylim = c(1, 18), 
                  clip = 'off') +   # This keeps the labels from disappearing
  xlab('HR') + ylab('Multivariate Cox regression')+theme_bw()+theme(panel.grid=element_blank())+
  theme(legend.position = 'None',plot.margin = unit(c(1,8,1,1), "lines"),
        legend.title = element_text( family = 'Arial_a', size = 7), legend.text = element_text( family = 'Arial_a', size = 7), axis.text.y = element_text(hjust = 1, family = 'Arial_a', size = 7, colour = 'black'), axis.text.x = element_text(family = 'Arial_a', size = 7, colour = 'black'), axis.title.x = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = -1), axis.title.y = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = 1))+
  geom_text(aes(x = 600, label=result_na$p), color = 'black',hjust=1, size = 7/.pt, family = 'Arial_a')+
  geom_text(aes(x = 40, label=result_na$`exp(coef) [confint]`), color = 'black',hjust=0, size = 7/.pt, family = 'Arial_a')+
  annotate("text", x = 43 , y = -0.5,label = "HR (95% CI)", color = 'black',hjust=0, size = 7/.pt, family = 'Arial_a')+
  annotate("text", x = 610 , y = -0.5,label = "p value", color = 'black',hjust=1, size = 7/.pt, family = 'Arial_a')+
  scale_x_continuous(limits= c(0.07, 10000), breaks= c(0.1, 0.2, 0.5, 2, 5, 10), trans = 'log10') +scale_color_my(palette = 'mixed')
pm_mul
#ggsave("/sf_unicox_inner.pdf",p_cox, width=400/72, height=270/72, dpi = 24)
#ggsave("/sf_mulcox_inner.pdf",pm_mul, width=400/72, height=270/72, dpi = 24)
