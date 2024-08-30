SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE p.Id = pl.PostId
  AND p.Id = ph.PostId
  AND p.Id = c.PostId
  AND u.Id = c.UserId
  AND u.Id = v.UserId
  AND c.Score = 0
  AND c.CreationDate >= CAST('2010-08-02 20:27:48' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-10 16:09:23' AS timestamp)
  AND p.PostTypeId = 1
  AND p.Score = 4
  AND p.ViewCount <= 4937
  AND pl.CreationDate >= CAST('2011-11-03 05:09:35' AS timestamp)
  AND ph.PostHistoryTypeId = 1
  AND u.Reputation <= 270
  AND u.Views >= 0
  AND u.Views <= 51
  AND u.DownVotes >= 0;
