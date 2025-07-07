library(survminer)
library(survival) 

library(wesanderson)
library(showtext) 
font_add("Arial_a", '/Library/Fonts/Arial Unicode.ttf')

pal <- wes_palette("Zissou1", 2, type = "continuous")

my_colors <- c(
  `purple` = "#CF3C48",
  `yellow` = "#045D94")
my_cols <- function(...){
  cols <- c(...)
  if (is.null(cols))
    return(my_colors)
  my_colors[cols]
}
my_palettes <- list(
  `main` = my_cols("purple","yellow"))

my_pal <- function(palette="main", reverse=FALSE, ...){
  pal <- my_palettes[[palette]]
  if (reverse) pal <- rev(pal)
  colorRampPalette(pal, ...)
}

scale_color_my <- function(palette="main", discrete=TRUE, reverse=FALSE, ...){
  pal <- my_pal(palette = palette, reverse = reverse)
  if (discrete){
    discrete_scale("colour", paste0("my_", palette), palette = pal, ...)
  }else{
    scale_color_gradientn(colours = pal(256), ...)
  }
}
csv = read.table('/data_all_tj.csv', sep = ',', header = TRUE)
#csv = csv[csv$strat == 0,]
strat = csv$strat
time = csv$time
censor = csv$censor
chemo = csv$adj_after
csv_ccc = data.frame('strat'=strat, 'time'=time, 'censor'=censor, 'chemo' = chemo)
fit <- survfit(Surv(time,censor) ~ strat,  
               data = csv_ccc)
aaa <- survdiff(Surv(time,censor) ~ strat, 
               data = csv_ccc)

p_r = surv_pvalue(fit)$pval
mystars <- ifelse(p_r <  .0001, '****', ifelse(p_r < .001, "***", ifelse(p_r < .01, "**", ifelse(p_r < .05, "*", "ns, "))))
p_v = ifelse(p_r<0.0001, 'p < 0.0001', surv_pvalue(fit)$pval.txt)
p_v = paste(mystars, p_v, sep = "")
fit
summary(fit)
p = ggsurvplot(fit, data = csv,
               size = 0.7, 
           xlab = 'Follow-up time (months)',
           ylab = 'Relapse-free survival',
           legend.title = "",
           censor = TRUE,
           legend.labs = c("No ACT", "ACT"),
           palette = pal,
           break.time.by = 20, 
           ggtheme = theme_classic2(base_family = 'Arial_a'),
           font.family = 'Arial_a',
           risk.table = T,
           risk.table.fontsize = 7/.pt,
           risk.table.y.text = T,
           risk.table.y.text.col = FALSE,
           risk.table.pos = 'out',
           risk.table.height = 0.5,
           font.main = c(7,'bold'),
           font.title= c(7),
           font.subtitle= c(7),
           font.caption= c(7),
           font.x = c(7, 'bold'),
           font.y = c(7, 'bold'),
           font.tickslab = c(7,'bold'),
           font.legend = c(7),
           tables.theme = theme_cleantable() +
  theme(plot.margin = unit(c(1,1,1,10),'mm'),plot.title = element_text(hjust = -0.4,size = 8, face = 'bold',family = 'Arial_a'), axis.title.y = element_text(hjust=-1,family = 'Arial_a', color = 'gray'),axis.text = element_text(hjust = 0, size = 7, color = 'black',family = 'Arial_a'))
  ) %++% coord_cartesian(xlim=c(0, 100))
p$plot = p$plot + annotate("text", x = 20, y = 0.2, label = p_v, size = 7/.pt,family = 'Arial_a')+
  scale_color_my(palette = "main")+

  theme(axis.line = element_line(size = 0.5),legend.direction = 'horizontal',legend.position = 'top', legend.key.height=unit(0, "cm"), legend.spacing.y = unit(0, 'mm'),legend.text = element_text(size = 7), title = element_text(size = 7), axis.text.x = element_text(size = 7, colour = 'black'), axis.text.y = element_text(hjust = -3, size = 7, colour = 'black'), axis.title.x = element_text(size = 8, face = 'bold', vjust = 0), axis.title.y = element_text(size = 8, face = 'bold', vjust = 0))+
  scale_size_manual(values = c(0.1, 0.1))
p
p$table = p$table + theme(text = element_text(family = 'Arial_a'))

library(patchwork)

pp = (p$plot / p$table) + plot_layout(heights = c(4, 1))
grid.draw.ggsurvplot <- function(x){
  survminer:::print.ggsurvplot(x, newpage = FALSE)
}
pp
#ggsave("/chemo_kmcurv_tj.pdf",plot=pp, width=300/72, height=300/72, dpi = 24, units = 'in')

