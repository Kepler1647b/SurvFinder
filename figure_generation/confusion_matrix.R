#######size 386 300
library(ggplot2)
library(showtext)
font_add("Arial_a", '/Library/Fonts/Arial Unicode.ttf')
csv = read.table('/data_all_inner.csv', sep = ',', header = TRUE)
csv$label = csv$censor
csv$value = csv$pvalue
threshold = 0.088
csv = csv[csv$include != 0,]
tb = table(csv$label, as.integer(csv$value > threshold))
SurvFinder = factor(c(0,0,1,1))
RFS = factor(c(1,0,1,0))
Y_per_o = as.double(prop.table(tb,1))
Y_per_o[Y_per_o>0.8] = 0.8
Y_per_o[Y_per_o<0.2] = 0.2
Y_per = Y_per_o * 100
df <- data.frame(SurvFinder, RFS, Y_per)
Y = as.double(tb)
library(wesanderson)
pal <- wes_palette("Zissou1", 50, type = "continuous")
pal = c('#045D94', "#7FADC9", '#EBDAAD', "#F2CBAD", '#CF3C48')

library(ggplot2)
p_cm = ggplot(df, aes(SurvFinder, RFS, fill = Y_per), size = 4) +
  geom_tile(aes(fill = Y_per, width=.98, height=.98), colour = "white") +
  geom_text(size = 8/.pt, aes(label = sprintf("%1.0f", Y)), colour = 'black', fontface = 'bold', family = 'Arial_a') +
  scale_fill_gradientn(colours = pal, limits = c(20,80), breaks = c(30, 50, 70), guide = 'colourbar') +
  guides(fill = guide_colorbar(title = '(%) of \nrow \ntotal'))+
  labs(x = 'Prediction', y = 'RFS')+
  theme_bw()+scale_y_discrete(labels=c('Relapse', 'No relapse'))+scale_x_discrete(labels=c('No relapse', 'Relapse'))+
  theme(panel.border = element_rect(color = "black", size = 1, fill = NA), legend.position = 'right',legend.title = element_text( family = 'Arial', size = 7), legend.text = element_text( family = 'Arial', size = 7), axis.text.y = element_text(angle = 90, hjust = 0.5, family = 'Arial', size = 7, colour = 'black'), axis.text.x = element_text(family = 'Arial', size = 7, colour = 'black'), axis.title.x = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust = -1), axis.title.y = element_text(family = 'Arial_a', size = 8, face = 'bold', vjust =1),
        legend.key.height = unit(0.5, "cm"), legend.key.width = unit(0.5, 'cm'))
ggsave("/wsinet_cm_inner.pdf",p_cm, width=200/72, height=150/72, dpi = 24)
p_cm