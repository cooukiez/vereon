cd ./src/shader/glsl || exit
#glslangValidator -e main -gVS -S vert -Os -V "shader.vert" -o "vert.spv"
#glslangValidator -e main -gVS -S frag -Os -V "shader.frag" -o "frag.spv"
glslangValidator -e main -S vert -Os -V "shader.vert" -o "vert.spv"
glslangValidator -e main -S frag -Os -V "shader.frag" -o "frag.spv"