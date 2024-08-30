SELECT COUNT(*)
FROM comments AS c, posts AS p, postLinks AS pl
WHERE c.UserId = p.OwnerUserId
  AND p.Id = pl.PostId
  AND c.Score = 0
  AND p.CreationDate >= CAST('2010-09-06 00:58:21' AS timestamp)
  AND p.CreationDate <= CAST('2014-09-12 10:02:21' AS timestamp)
  AND pl.LinkTypeId = 1
  AND pl.CreationDate >= CAST('2011-07-09 22:35:44' AS timestamp);
