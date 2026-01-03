using MySQL
using DBInterface
using DataFrames

conn = DBInterface.connect(MySQL.Connection, "localhost", "itsame", "idontknow", db="", port=3306)

#prep = DBInterface.prepare(conn, sql)
#query = "
#select *
#from table;"
#DBInterface.execute(prep, query)
