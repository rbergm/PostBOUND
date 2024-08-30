SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v
WHERE p.Id = pl.PostId
  AND p.Id = v.PostId
  AND p.Id = ph.PostId
  AND p.Id = c.PostId
  AND c.Score = 0
  AND p.FavoriteCount >= 0
  AND p.CreationDate >= CAST('2010-07-23 02:00:12' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-08 13:52:41' AS timestamp)
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-10-06 21:41:26' AS timestamp)
  AND v.VoteTypeId = 2;
