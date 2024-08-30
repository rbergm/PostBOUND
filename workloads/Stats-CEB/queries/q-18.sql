SELECT COUNT(*)
FROM posts AS p, postLinks AS pl, postHistory AS ph
WHERE p.Id = pl.PostId
  AND pl.PostId = ph.PostId
  AND p.CreationDate >= CAST('2010-07-19 20:08:37' AS timestamp)
  AND ph.CreationDate >= CAST('2010-07-20 00:30:00' AS timestamp);
