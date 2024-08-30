SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v
WHERE p.Id = c.PostId
  AND p.Id = pl.PostId
  AND p.Id = ph.PostId
  AND p.Id = v.PostId
  AND c.CreationDate >= CAST('2010-08-01 12:12:41' AS timestamp)
  AND p.Score <= 44
  AND p.FavoriteCount >= 0
  AND p.FavoriteCount <= 3
  AND p.CreationDate >= CAST('2010-08-11 13:53:56' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-03 11:52:36' AS timestamp)
  AND pl.LinkTypeId = 1
  AND pl.CreationDate <= CAST('2014-08-11 17:26:31' AS timestamp)
  AND ph.CreationDate >= CAST('2010-09-20 19:11:45' AS timestamp)
  AND v.CreationDate <= CAST('2014-09-11 00:00:00' AS timestamp);
