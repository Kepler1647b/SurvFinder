###size 350
library(pROC)
library(ROCR)
library(multiROC)
library(ggplot2)
library(stringr)
library(wesanderson)
library(showtext)
library(extrafont)
font_add("Arial_a", '/Library/Fonts/Arial Unicode.ttf')
showtext_auto() # 后面字体均可以使用导入的字体
pal <- wes_palette("Zissou1", 5, type = 'continuous')
pal = c('#045D94', "#7FADC9", '#EBDAAD', "#F2CBAD", '#CF3C48')
roc_with_ci <- function(obj) {
  ciobj <- ci.se(obj, specificities = seq(0, 1, l = 25))
  dat.ci <- data.frame(x = as.numeric(rownames(ciobj)),
                       lower = ciobj[, 1],
                       upper = ciobj[, 3])
  auctext = str_c('Mean ROC (AUC = ', round(obj[[9]][1],3), ')')
  obj_l = list('auc'=obj)
  ci.list <- lapply(obj_l, ci.se, specificities = seq(0, 1, l = 25))
  dat.ci.list <- lapply(ci.list, function(ciobj) 
    data.frame(x = as.numeric(rownames(ciobj)),
               lower = ciobj[, 1],
               upper = ciobj[, 3]))
  #dev.new(width=0.1, height=0.1)
  p <- ggroc(obj_l,legacy.axes = TRUE, size = 0.7)
  
  p <- p + geom_ribbon(
    data = dat.ci.list[[1]],
    aes(x = 1-x, ymin = lower, ymax = upper, fill ="± 1 std. dev."),
    alpha = 0.1,
    inherit.aes = F) 
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
    scale_fill_manual(NULL, values = pal[5]) +
      scale_color_manual(labels = c(auctext), values = pal[5]) +
      guides(
        color=guide_legend(override.aes = list(fill=NA), order=1, title = ''),
        fill=guide_legend(override.aes = list(color=NA), order=2, keyheight = 0.2, title = '')
      )+
      theme(legend.position = c(0.66,0.15), legend.spacing.y = unit(-3, 'mm'), legend.title.position = 'left',legend.text = element_text( family = 'Arial', size = 7), title = element_text(family = 'Arial', size = 7), axis.text.x = element_text(family = 'Arial', size = 7, colour = 'black'), axis.text.y = element_text(family = 'Arial', size = 7, colour = 'black'), axis.title.x = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = -1), axis.title.y = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = 1))
  p
  #ggsave("/mvnet_roc_tcga.pdf",p, width=200/72, height=200/72, dpi = 24)
} 

csv = read.table('/ensemble_fusion_tcga.csv', sep = ',', header = TRUE)
S0_true = as.integer(csv$label == 0)
S1_true = as.integer(csv$label == 1)
rocobj0 <- roc(S1_true, csv$mean,auc = TRUE,
               ci=TRUE, # compute AUC (of AUC by default)
               print.auc=TRUE) # print the AUC (will contain the CI)
rocobj
auctext = str_c('Mean ROC (AUC = ', round(rocobj0[[9]][1],3), ' )')
roc_with_ci(rocobj0)

