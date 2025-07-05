### size 400
library(pROC)
library(ROCR)
library(multiROC)
library(ggplot2)
library(ggforce)
library(ggpubr)

library(wesanderson)
library(showtext) 
font_add("Arial_a", '/Library/Fonts/Arial Unicode.ttf')
pal <- wes_palette("Zissou1", 3, type = "continuous")
pal = c('#045D94', '#EBDAAD', '#CF3C48')

roc_with_ci <- function(obj,obj1,obj2) {
  
  auctext0 = str_c('TLS-like ROC (AUC = ', round(obj[[9]][1],4), ')')
  auctext1 = str_c('Normal ROC (AUC = ', round(obj1[[9]][1],4), ')')
  auctext2 = str_c('Tumor ROC (AUC = ', round(obj2[[9]][1],4), ')')
  
  obj1[[9]][1]
  obj_l = list('value0'=obj, 'value1'=obj1, 'value2'=obj2)

  p <- ggroc(obj_l,legacy.axes = TRUE, size = 0.7)
  
  p = p+
    theme_classic() +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      alpha = 0.7,
      color = "black"
    ) + coord_equal() + 
    labs(x = "1 - Specificity", y = "Sensitivity")+
    scale_color_manual(labels = c(auctext0, auctext1, auctext2), values = pal) +
    guides(
      color=guide_legend(override.aes = list(fill=NA), order=1, title = '', keyheight = 0.6),
      fill=guide_legend(override.aes = list(color=NA), order=2, keyheight = 0.2, title = '')
    )+
    theme(legend.position = c(0.63,0.12), legend.spacing.y = unit(-5, 'mm'), legend.title.position = 'left',legend.text = element_text( family = 'Arial_a', size = 7), title = element_text(family = 'Arial_a', size = 7), axis.text.x = element_text(family = 'Arial_a', size = 7, colour = 'black'), axis.text.y = element_text(family = 'Arial_a', size = 7, colour = 'black'), axis.title.x = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = -1), axis.title.y = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = 1))
    
  p1 <- ggroc(obj_l,legacy.axes = TRUE, size = 0.5)
  
  p1 = p1+
    theme_classic() +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      alpha = 0.7,
      color = "black"
    ) + coord_equal() + 
    labs(x = "1 - Specificity", y = "Sensitivity")+
    scale_color_manual(labels = c(auctext0, auctext1, auctext2), values = pal) +
    guides(
      color=guide_legend(override.aes = list(fill=NA), order=1, title = '', keyheight = 0.6),
      fill=guide_legend(override.aes = list(color=NA), order=2, keyheight = 0.2, title = '')
    )+
    theme(legend.position = c(0.63,0.15), legend.spacing.y = unit(-5, 'mm'), legend.title.position = 'left',legend.text = element_text( family = 'Arial_a', size = 7), title = element_text(family = 'Arial_a', size = 7), axis.text.x = element_text(family = 'Arial_a', size = 7, colour = 'black'), axis.text.y = element_text(family = 'Arial_a', size = 7, colour = 'black'), axis.title.x = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = -1), axis.title.y = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = 1))

    p2 = p1 + 
    xlim (-0.025, 0.05) +
    ylim (0.95, 1.025) +
    labs(x = '', y = '')+
    theme_pubr( base_size = 8,border = TRUE)+
    theme(legend.position = 'None', plot.background = element_rect(fill='transparent'), axis.text.x = element_text(family = 'Arial_a', size = 7, colour = 'black'), axis.text.y = element_text(family = 'Arial_a', size = 7, colour = 'black'))
  #p2
  p3 = p + 
    annotation_custom(ggplotGrob(p2), xmin = 0.2, xmax = 1, ymin = 0.1, ymax = 0.9) +
    geom_rect(aes(xmin = -0.05, xmax = 0.1, ymin = 0.9, ymax = 1.05), color='black', linetype='dashed', alpha=0) 
  ggsave("/ntl_roc_inner.pdf",p3, width=200/72, height=200/72, dpi = 24)
  p3
} 
csv = read.table('/result_ntl_inner_finalpt.csv', sep = ',', header = TRUE)
S0_true = as.integer(csv$label == 0)
S1_true = as.integer(csv$label == 1)
S2_true = as.integer(csv$label == 2)
rocobj0 <- roc(S0_true, csv$value0,auc = TRUE,
               ci=TRUE, # compute AUC (of AUC by default)
               print.auc=TRUE) # print the AUC (will contain the CI)
rocobj1 <- roc(S1_true, csv$value1,auc = TRUE,
               ci=TRUE, # compute AUC (of AUC by default)
               print.auc=TRUE) # print the AUC (will contain the CI)
rocobj2 <- roc(S2_true, csv$value2,auc = TRUE,
               ci=TRUE, # compute AUC (of AUC by default)
               print.auc=TRUE) # print the AUC (will contain the CI)
rocobj
auctext = str_c('Mean ROC (AUC = ', round(rocobj0[[9]][1],4), ' )')
roc_with_ci(rocobj0, rocobj1, rocobj2)

