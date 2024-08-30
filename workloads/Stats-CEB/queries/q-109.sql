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
  AND pl.CreationDate >= CAST('2011-11-21 22:39:41' AS timestamp)
  AND pl.CreationDate <= CAST('2014-09-01 16:29:56' AS timestamp)
  AND v.CreationDate >= CAST('2010-07-22 00:00:00' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-14 00:00:00' AS timestamp);
