using MySQL
using DBInterface
using DataFrames

function getDbData(query)
  conn = DBInterface.connect(MySQL.Connection, "crossover.proxy.rlwy.net", "root", "tLeGwnVSAGVQLYbaMcdSJLunevRZRghw", db="railway", port=17837)

  queryResults = DataFrame(DBInterface.execute(conn, query))

  return queryResults
end
