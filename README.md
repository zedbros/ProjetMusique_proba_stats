# ProjetMusique_proba_stats
Projet for probabilities and statistiques at HES-SO Valais-Wallis.

# Run
### Using Julia VS Code extension (recommended)
Simply locate the file and run the file via the small play button at the top right of the screen.

### Using native terminal



# Packages
## Package Mangement
This project uses the VS Code Julia extension. All the packages can be found in the @packageManager.jl file.
## Updating packages
##### Code
If additionnal packages are required, simply add the needed package as shown in the code.
To finish, simply run:<br>
``path> julia .\packageManager.jl``<br>
(on windows) to update the local packages.
##### CMD
It is also possible to manage packages through the command line:

``julia> ]``<br>
``pgk> activate .``<br>
``pkg> instantiate``<br>
``add/remove/update CSV``<br>

for example.

## View currently installed packages
Either look in the ``Project.toml`` file or ``pkg> status`` in terminal.
