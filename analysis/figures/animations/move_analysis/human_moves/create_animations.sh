for folder in directory
convert -delay 50 -loop 0 folder/*.svg folder.gif
endfor