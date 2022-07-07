#author: "Evan Brown & Daniel Craig"
#date: "7/7/2022"
#purpose: Render project2.Rmd as a .md file called README.md for our Project 2 repo.


rmarkdown::render(
  input="project2.Rmd",
  output_format = "github_document",
  output_file = "README.md",
  runtime = "static",
  clean = TRUE,
  params = NULL,
  knit_meta = NULL,
  envir = parent.frame(),
  run_pandoc = TRUE,
  quiet = FALSE,
  encoding = "UTF-8"
)