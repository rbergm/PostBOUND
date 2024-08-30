SELECT COUNT(*)
FROM comments AS c,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  posts AS p
WHERE pl.PostId = p.Id
  AND c.PostId = p.Id
  AND v.PostId = p.Id
  AND ph.PostId = p.Id
  AND c.Score = 0
  AND pl.CreationDate >= CAST('2011-03-22 06:18:34' AS timestamp)
  AND pl.CreationDate <= CAST('2014-08-22 20:04:25' AS timestamp);
