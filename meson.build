project(
  'nbody_gui',
  'cpp',
  version : '0.1',
  default_options : [
    'warning_level=3', 
    'b_sanitize=address',
    'debug=true',
    'buildtype=debugoptimized', 
    'cpp_std=c++11',
    'werror=true',
  ]
)

cc = meson.get_compiler('cpp')

add_global_arguments(language : 'cpp')

m_dep = cc.find_library('m', required : false)

sfml_sys = dependency('sfml-system')
sfml_win = dependency('sfml-window')
sfml_graph = dependency('sfml-graphics')

opengl = dependency('GL')

inc = include_directories('src/include', 'src/include/imgui')

subdir('src')
